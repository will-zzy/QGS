/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

#include "stopthepop_QGS/stopthepop_common.cuh"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	float2* rects,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	// Generate no key/value pair for invisible Gaussians

	// QGS' reminder:
	// Is it possible for a Gaussian to have radii > 0 while tiles_touched = 0? No, it’s not possible: if tiles_touched = 0, 
	// radii would not be assigned a value in preprocessCUDA. 
	// However, it is possible for the Gaussian to have radii = 0 while tiles_touched \neq 0 (The issue originates from 
	// rounding errors in the forced type conversion in getRect). In this case, the code would not enter this if branch, 
	// and tile would not be assigned a valid tile ID | depth. As a result, the tile ID becomes a random value, 
	// leading to an out-of-bounds error.

	// This is typically more common in 2DGS/QGS, as being surface representations makes it more likely for the bounding box 
	// radius to be zero. In contrast, for 3DGS, achieving radius = 0 requires the eigenvalues of the covariance matrix to be 
	// zero, which is a much stricter condition.
	if (radii[idx] > 0) {
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);
		int tiles_touched = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
		int tiles_sort = (idx == 0) ? offsets[idx] : offsets[idx] - offsets[idx - 1];
		if (tiles_sort != tiles_touched){
			printf("tiles_touched:%d\n",tiles_touched);
			printf("tiles_sort:%d\n\n",tiles_sort);
		}
		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}


// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges, int total_tiles_num)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	uint32_t depth = static_cast<uint32_t>(key & 0xFFFFFFFF);

	// If this if branch is entered, it indicates that some Gaussian primitives have not been assigned a key, 
	// causing currtile to be a random value. This is usually due to a Gaussian with an excessively small scale, 
	// resulting in radii = 0.
	if (currtile >= total_tiles_num){
		printf("currtile_ID:%u, depth:%f\n", currtile, *(float*)&depth);
	}
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.rects2D, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.view2gaussian, P * 16, 128);
	obtain(chunk, geom.rscales_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.scales_sign, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::PointState CudaRasterizer::PointState::fromChunk(char*& chunk, size_t P)
{
	PointState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.points2D, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 6, 128); // T, dist1, dist2, distortion_before_normalized, curv1, curv2
	obtain(chunk, img.center_depth, N, 128);
	obtain(chunk, img.center_alphas, N, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.point_ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;

}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	float* aabb,
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float sigma,
	const float* rotations,
	const float* view2gaussian_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float* cam_intr,
	const float kernel_size,
	const float* subpixel_offset,
	const bool prefiltered,
	const bool return_depth,
	const bool return_normal,
	float* out_colors,
	int* n_touched,
	int* radii,
	bool debug)
{
	const float focal_x = cam_intr[0];
	const float focal_y = cam_intr[1];
	const float principal_x = cam_intr[2];
	const float principal_y = cam_intr[3];

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		aabb,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		sigma,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		colors_precomp,
		view2gaussian_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		principal_x, principal_y,
		kernel_size,
		radii,
		geomState.rects2D,
		geomState.means2D,
		geomState.depths,
		geomState.view2gaussian,
		geomState.rgb,
		geomState.rscales_opacity,
		geomState.scales_sign,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
#if TILE_SORTING
    FORWARD::duplicate(
	P, width, height, focal_x, focal_y, principal_x, principal_y, sigma,
	geomState.means2D,
	geomState.depths,
	geomState.rscales_opacity,
	(float3*)scales,
	geomState.scales_sign,
	geomState.view2gaussian,
	geomState.point_offsets,
	radii,
	geomState.rects2D,
	binningState.point_list_keys_unsorted,
	binningState.point_list_unsorted,
	tile_grid
);
#else
	
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		geomState.rects2D,
		tile_grid)
#endif

	
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges,
			tile_grid.x * tile_grid.y);
	CHECK_CUDA(, debug)
	// printf("ok2\n");
	
	//printf("in CudaRasterizer::Rasterizer::forward, P: %d num_rendered: %d geo_chunk_size: %d img_chunk_size: %d, binning_chunk_size: %d\n", P, num_rendered, chunk_size, img_chunk_size, binning_chunk_size);
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* view2gaussian = view2gaussian_precomp != nullptr ? view2gaussian_precomp : geomState.view2gaussian;
	// const float* view2gaussian = view2gaussian_precomp;
	CHECK_CUDA(FORWARD::render(
		P,
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		principal_x, principal_y,
		sigma,
		feature_ptr,
		view2gaussian,
		viewmatrix,
		(float3*)means3D,
		(float3*)scales,
		geomState.depths,
		geomState.rscales_opacity,
		geomState.scales_sign,
		return_depth,
		return_normal,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_colors,
		n_touched), debug)

	return num_rendered;
}


// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* view2gaussian_precomp,
	const float* scales,
	const float scale_modifier,
	const float sigma,
	const bool stop_z_gradient,
	const bool reciprocal_z,
	const float* rotations,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float* cam_intr,
	const float tan_fovx, float tan_fovy,
	const float kernel_size,
	const float* subpixel_offset,
	const int* radii,
	const float* out_colors,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dview2gaussian,
	const bool return_depth,
	const bool return_normal,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_x = cam_intr[0];
	const float focal_y = cam_intr[1];
	const float principal_x = cam_intr[2];
	const float principal_y = cam_intr[3];

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	
	const float* view2gaussian_ptr = (view2gaussian_precomp != nullptr) ? view2gaussian_precomp : geomState.view2gaussian;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		principal_x, principal_y,
		sigma,
		stop_z_gradient,
		background,
		geomState.rscales_opacity,
		geomState.scales_sign,
		color_ptr,
		view2gaussian_ptr,
		viewmatrix,
		(float3*)means3D,
		(float3*)scales,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		out_colors,
		dL_dpix,
		return_depth,
		return_normal,
		dL_dopacity,
		dL_dcolor,
		dL_dscale,
	    dL_dview2gaussian), debug)

	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		width, height,
		reciprocal_z,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		focal_x, focal_y,
		scale_modifier,
		view2gaussian_ptr,
		viewmatrix,
		projmatrix,
		kernel_size,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dview2gaussian,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		dL_dopacity), debug)
}
