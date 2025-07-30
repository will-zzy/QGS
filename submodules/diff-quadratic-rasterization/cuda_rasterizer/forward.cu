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

#include "forward.h"
#include "auxiliary.h"
#include "stopthepop_QGS/stopthepop_common.cuh"
#include "stopthepop_QGS/resorted_render.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}


// TODO combined with computeCov3D to avoid redundant computation
// Forward method for creating a view to gaussian coordinate system transformation matrix
__device__ void computeView2Gaussian(const float3& mean, const glm::vec4 rot, const float* viewmatrix, float* view2gaussian, glm::mat4& G2V)
{
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// transform 3D points in gaussian coordinate system to world coordinate system as follows
	// new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
	// so the rots is the gaussian to world transform

	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f
	);

	// could be simplied by using pointer
	// viewmatrix is the world to view transformation matrix
	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);

	// Gaussian to view transform
	G2V = W2V * G2W;
	glm::mat3 R_transpose = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);

	glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	glm::vec3 t2 = -R_transpose * t;

	// Compute inverse of G2V
	view2gaussian[0] = R_transpose[0][0];
	view2gaussian[1] = R_transpose[0][1];
	view2gaussian[2] = R_transpose[0][2];
	view2gaussian[3] = 0.0f;
	view2gaussian[4] = R_transpose[1][0];
	view2gaussian[5] = R_transpose[1][1];
	view2gaussian[6] = R_transpose[1][2];
	view2gaussian[7] = 0.0f;
	view2gaussian[8] = R_transpose[2][0];
	view2gaussian[9] = R_transpose[2][1];
	view2gaussian[10] = R_transpose[2][2];
	view2gaussian[11] = 0.0f;
	view2gaussian[12] = t2.x;
	view2gaussian[13] = t2.y;
	view2gaussian[14] = t2.z;
	view2gaussian[15] = 1.0f;
}


// 4*a*\rho^2 + 6*\rho - 7*sigma = 0
__device__ float GetRootfromEquation(const float sigma, const float a){
	if (a < 5e-3)
		return sigma; // when \rho \in [0, 30], error is within 1.5%
	float r2A = __frcp_rn(2 * 4 * a);
	float B2_4AC = 36.00000f + 112.00000f * a * sigma;
	return (-6.00000f + sqrt(B2_4AC)) * r2A; 
}



__device__ bool computeAABB(
	const glm::mat4 G2V, 
	const glm::vec3 scale,
	const float sigma, // [0.0, 3.0]
	const float4 intrins,
	float2 & center, 
	float2 & extent,
	float3 & rscale,
	int3* s_sign
){
	rscale.x = __frcp_rn(scale.x * scale.x);
	rscale.y = __frcp_rn(scale.y * scale.y);
	rscale.z = __frcp_rn(scale.z);

	float s_max = max(abs(scale.x), abs(scale.y));
	const float rs2_max = min(rscale.x, rscale.y);
	
	s_sign->x = copysign(1, scale.x);
    s_sign->y = copysign(1, scale.y);
    s_sign->z = copysign(1, scale.z);

	// Approximate the circular paraboloid using its major-axis representation.
	const float a0 = s_sign->z * scale.z * rscale.x;
	const float a1 = s_sign->z * scale.z * rscale.y;
	// find x0 S.t. f(ax0^2) = sigma * s_max, see Eq. 23 in the paper of QGS
	const float l0 = GetRootfromEquation(sigma * abs(scale.x), a0); 
	const float l1 = GetRootfromEquation(sigma * abs(scale.y), a1); 

	// intersected line (i.e. x = x_0, y = x_1) of tagent plane of each axis and plane z=0 
	float x_0 = (l0 * abs(scale.x) - l0 * l0 / 2) / abs(scale.x);
	float x_1 = (l1 * abs(scale.y) - l1 * l1 / 2) / abs(scale.y);
	
	x_0 = min(l0, max(x_0, 0.0f));
	x_1 = min(l1, max(x_1, 0.0f));
	//

	const float l_2 = abs(scale.x) > abs(scale.y) ? a0 * l0 * l0 : a1 * l1 * l1;

	const bool saddle = (s_sign->x * s_sign->y) < 0 ? true: false;
	const bool convex = (!saddle) && (s_sign->x * s_sign->z >= 0) ? true : false;
	const bool concave = (!saddle) && (s_sign->x * s_sign->z < 0) ? true : false;

	float Points[24] = {0.0f}; // 3D BBox
	float l_2_divide_10 = l_2 / 10;
	if(convex){
		Points[0] = -x_0; 
		Points[1] = -x_1;
		Points[2] = -l_2_divide_10;
		
		Points[3] = x_0;  
		Points[4] = -x_1;
		Points[5] = -l_2_divide_10;
		
		Points[6] = -x_0;
		Points[7] = x_1;
		Points[8] = -l_2_divide_10;
		
		Points[9] = x_0;
		Points[10] = x_1;
		Points[11] = -l_2_divide_10;
		
		Points[12] = -l0;
		Points[13] = -l1;
		Points[14] = l_2;
		
		Points[15] = l0;
		Points[16] = -l1;
		Points[17] = l_2;

		Points[18] = -l0;
		Points[19] = l1;
		Points[20] = l_2;
		
		Points[21] = l0;
		Points[22] = l1;
		Points[23] = l_2;
	}
	else if(concave){
		Points[0] = -l0; 
		Points[1] = -l1;
		Points[2] = -l_2;
		
		Points[3] = l0;  
		Points[4] = -l1;
		Points[5] = -l_2;
		
		Points[6] = -l0;
		Points[7] = l1;
		Points[8] = -l_2;
		
		Points[9] = l0;
		Points[10] = l1;
		Points[11] = -l_2;
		
		Points[12] = -x_0;
		Points[13] = -x_1;
		Points[14] = l_2_divide_10;
		
		Points[15] = x_0;
		Points[16] = -x_1;
		Points[17] = l_2_divide_10;

		Points[18] = -x_0;
		Points[19] = x_1;
		Points[20] = l_2_divide_10;
		
		Points[21] = x_0;
		Points[22] = x_1;
		Points[23] = l_2_divide_10;
	}
	else{
		Points[0] = -l0; 
		Points[1] = -l1;
		Points[2] = -l_2;
		
		Points[3] = l0;  
		Points[4] = -l1;
		Points[5] = -l_2;
		
		Points[6] = -l0;
		Points[7] = l1;
		Points[8] = -l_2;
		
		Points[9] = l0;
		Points[10] = l1;
		Points[11] = -l_2;
		
		Points[12] = -l0;
		Points[13] = -l1;
		Points[14] = l_2;
		
		Points[15] = l0;
		Points[16] = -l1;
		Points[17] = l_2;

		Points[18] = -l0;
		Points[19] = l1;
		Points[20] = l_2;
		
		Points[21] = l0;
		Points[22] = l1;
		Points[23] = l_2;
	}



	float x_min = FLT_MAX;
	float y_min = FLT_MAX;

	float x_max = -FLT_MAX;
	float y_max = -FLT_MAX;
	
	// 
	#pragma unroll
	for(int i = 0; i < 8; i++){
		// glm::vec4 P = glm::vec4((float)((i & 1) << 1) - 1, 
		// 			(float)(i & 2) - 1,
		// 			(float)((i & 4) >> 1) - 1,
		// 			1.0f);
		// P.z *= i & 4 ? (float)(saddle | convex) : (float)(saddle | concave);

		// P.x *= l0;
		// P.y *= l1;
		// P.z *= l_2;
		glm::vec4 P = {Points[3 * i + 0], Points[3 * i + 1], Points[3 * i + 2], 1.0f};
		
		glm::vec4 p_image = G2V * P;
		p_image.z = max(p_image.z, 0.01);

		// Divide by the focal length; otherwise, numerical overflow may occur in large-scale scenes.
		p_image.x = p_image.x / p_image.z + intrins.z / intrins.x;
		p_image.y = p_image.y / p_image.z + intrins.w / intrins.y;
		
		if(x_min > p_image.x)
			// x_min = intrins.x * p_image.x + intrins.z;
			x_min = p_image.x;
		if(x_max < p_image.x)
			// x_max = intrins.x * p_image.x + intrins.z;
			x_max = p_image.x;
		if(y_min > p_image.y)
			// y_min = intrins.y * p_image.y + intrins.w;
			y_min = p_image.y;
		if(y_max < p_image.y)
			// y_max = intrins.y * p_image.y + intrins.w;
			y_max = p_image.y;
	}

	center.x = 0.5 * (x_max + x_min);
	center.y = 0.5 * (y_max + y_min);

	// extent.x = ceil(max(0.0f, 0.5 * (x_max - x_min)));
	// extent.y = ceil(max(0.0f, 0.5 * (y_max - y_min)));
	extent.x = max(0.0f, 0.5 * (x_max - x_min));
	extent.y = max(0.0f, 0.5 * (y_max - y_min));
	// printf("center:%f,%f\nextent:%f,%f\n\n", center.x, center.y,extent.x,extent.y);
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	float* aabb,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const float sigma,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* view2gaussian_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float principal_x, float principal_y,
	const float kernel_size,
	int* radii,
	float2* rects,
	// uint2* rects,
	float2* points_xy_image,
	float* depths,
	float* view2gaussians,
	float* rgb,
	float4* rscales_opacity,
	int3* scales_sign,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{	
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;

	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// Transform point by projecting
	float3 p_view;
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	bool ok = true;
	const float* view2gaussian;
	float4 intrins = {focal_x, focal_y, principal_x, principal_y};
	glm::mat4 G2V;
	if (view2gaussian_precomp == nullptr)
	{
		// This is borrowed from GOF
		computeView2Gaussian(p_orig, rotations[idx], viewmatrix, view2gaussians + idx * 16, G2V);
		
	} else {
		view2gaussian = view2gaussian_precomp + idx * 16;
	}

	float2 center;
	float2 extent;
	float3 rscale;
	ok = computeAABB(G2V, scales[idx], sigma, intrins, center, extent, rscale, scales_sign + idx);	
	if(!ok)
		return;
	rscales_opacity[idx] = {rscale.x, rscale.y, rscale.z, opacities[idx]};

	// Filter out primitives whose bounding boxes lie entirely outside the image.
	double2 rect_min_div_focal = {center.x - extent.x, center.y - extent.y};
	double2 rect_max_div_focal = {center.x + extent.x, center.y + extent.y};
	if (rect_max_div_focal.x < 0 || rect_max_div_focal.y < 0)
		return;
	if (rect_min_div_focal.x * (double)(focal_x / 1) > W || rect_min_div_focal.y * (double)(focal_y / 1) > H)
		return;
	extent.x *= focal_x;
	extent.y *= focal_y;
	center.x *= focal_x;
	center.y *= focal_y;
	
	uint2 rect_min, rect_max;

	getRect(center, extent, rect_min, rect_max, grid);

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	rects[idx] = extent;
	points_xy_image[idx] = center;

	radii[idx] = max(ceil(extent.x), ceil(extent.y));

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

	// Be careful! When the radii equals zero, tiles_touched may still be nonzero.
	// if (radii[idx] < 1){ 
	// 	printf("Warning! A primitive with zero scale may occur!\n radii:%d\n",radii[idx]);
	// 	printf("x:%u, y:%u, tiles_touched:%d\n",extent.x, extent.y, tiles_touched[idx]);
	// 	radii[idx] = max(1, int(ceil(sqrt(tiles_touched[idx]))));
	// }
	aabb[4 * idx] = rect_min.x;
	aabb[4 * idx + 1] = rect_min.y;
	aabb[4 * idx + 2] = rect_max.x;
	aabb[4 * idx + 3] = rect_max.y;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const int P,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const float principal_x, const float principal_y,
	const float sigma,
	const float* __restrict__ features,
	const float* __restrict__ view2gaussian,
	const float* viewmatrix,
	const float3* __restrict__ means3D,
	const float3* __restrict__ scales,
	const float* __restrict__ depths,
	const float4* __restrict__ rscales_opacity,
	const int3* __restrict__ scales_sign,
	const bool return_depth,
	const bool return_normal,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int* __restrict__ n_touched)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// create the direction
	float2 ray = { (pixf.x - principal_x) / focal_x, (pixf.y - principal_y) / focal_y };
	float3 ray_point = { ray.x , ray.y, 1.0 };

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	__shared__ float3 collected_scales[BLOCK_SIZE];
	__shared__ float4 collected_rscales_opacity[BLOCK_SIZE];
	__shared__ int3 collected_scales_sign[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t median_contributor = -1;
	float C[OUTPUT_CHANNELS] = { 0 };

	// float distortion = {0};
	float A = 0;
	float dist1 = {0};
	float dist2 = {0};
	float curv1 = {0};
	float curv2 = {0};
	float median_weight = {0};
	float curvature_sum = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_rscales_opacity[block.thread_rank()] = rscales_opacity[coll_id];
			collected_scales[block.thread_rank()] = scales[coll_id];
			collected_scales_sign[block.thread_rank()] = scales_sign[coll_id];
			for (int ii = 0; ii < 16; ii++)
				collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
			
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			const float* view2gaussian_j = collected_view2gaussian + j * 16;
			const float3 scale_sign_j = {(float)collected_scales_sign[j].x, (float)collected_scales_sign[j].y, (float)collected_scales_sign[j].z}; // sign of scale
			const float3 scale_j = collected_scales[j]; // scale with sign
			const float4 rscale_o_j = collected_rscales_opacity[j]; // {1/s1^2, 1/s2^2, 1/s3}
			const float3 rscale_sign_j = {rscale_o_j.x * scale_sign_j.x, rscale_o_j.y * scale_sign_j.y, rscale_o_j.z}; // {1/s1^2, 1/s2^2, 1/s3} with sign
			
			// create the ray
			float3 cam_pos_local = {view2gaussian_j[12], view2gaussian_j[13], view2gaussian_j[14]};
			float3 cam_ray_local = transformPoint4x3_without_t(ray_point, view2gaussian_j);

			// compute the roots of a quadratic equation.
			const double AA = rscale_sign_j.x * cam_ray_local.x * cam_ray_local.x + 
							  rscale_sign_j.y * cam_ray_local.y * cam_ray_local.y;

			const double BB = 2 * (rscale_sign_j.x * cam_pos_local.x * cam_ray_local.x + rscale_sign_j.y * cam_pos_local.y * cam_ray_local.y) - 
							  rscale_sign_j.z * cam_ray_local.z;

			const double CC = rscale_sign_j.x * cam_pos_local.x * cam_pos_local.x + 
							  rscale_sign_j.y * cam_pos_local.y * cam_pos_local.y - 
							  rscale_sign_j.z * cam_pos_local.z;

			float discriminant = BB*BB - 4*AA*CC;

			// If the discriminant is less than zero, the ray does not intersect the quadric.
			if(discriminant < 0)
				continue; 
			
			float r2AA = __frcp_rn(2 * AA);
			float discriminant_sq_r2AA = __fsqrt_rn(discriminant) * r2AA;

			// store the following variables for subsequent calculations.
			float root = 0.0f;
			float2 p = {0.0f, 0.0f};
			float p_norm_2 = 0.0f;
			float p_norm = 0.0f;
			float2 cos2_sin2 = {0.0f, 0.0f};
			float a = 0.0f;
			float s = 0.0f;
			float s_2 = 0.0f;
			float r0_2 = 0.0f;
			bool intersect = false;
			float sign = -99.0f;
			int AA_sign = copysign(1, AA);

			for(int i = -1; i < 2; i+=2){
#if	QUADRATIC_APPROXIMATION
				if (abs(AA) < 1e-6) // approximation of the intersection equation, see the supplementary material.
					root = - CC / BB;
				else
#endif				
				{
					sign = (float)i * (float)AA_sign;
					root = -BB * r2AA + sign * discriminant_sq_r2AA;
				}

				// see Equations (9), (10), and (12) in the QGS main text.
				p = {cam_pos_local.x + root * cam_ray_local.x, cam_pos_local.y + root * cam_ray_local.y};
				p_norm_2 = p.x * p.x + p.y * p.y + 1e-7;
				p_norm = __fsqrt_rn(p_norm_2);
				cos2_sin2 = {p.x * p.x / p_norm_2, p.y * p.y / p_norm_2};
				a = GetParabolaA(cos2_sin2, rscale_sign_j, scale_j);
				s = QuadraticCurveGeodesicDistanceOriginal(p_norm, a);
				s_2 = s * s;
				r0_2 = 1 / (cos2_sin2.x * rscale_o_j.x + cos2_sin2.y * rscale_o_j.y);

				// if the Gaussian weight at the intersection is too small, skip it.
				if (s_2 <= r0_2 * sigma * sigma / 1){
					intersect = true;
					break;
				}	
			}
			if (!intersect)
				continue;

			
			float power = - s_2 / (2 * r0_2);

			
			
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, rscale_o_j.w * exp(power));
			// float alpha = 1 * exp(-power * power);

			// float alpha = 0.99f;
			if (alpha < 1.0f / 255.0f)
				continue;
			alpha = 1 * exp(-sqrt(sqrt(-power)));
			done = true;
			float test_T = T * (1 - alpha);
			
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			if (return_depth){
				// NDC mapping is taken from 2DGS paper, please check here https://arxiv.org/pdf/2403.17888.pdf
				const float max_t = root;
				const float mapped_max_t = (FAR_PLANE * max_t - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * max_t);

				// distortion loss is taken from 2DGS paper, please check https://arxiv.org/pdf/2403.17888.pdf
				A = 1-T;
				float error = mapped_max_t * mapped_max_t * A + dist2 - 2 * mapped_max_t * dist1;
				C[DISTORTION_OFFSET] += error * alpha * T;
				
				dist1 += mapped_max_t * alpha * T;
				dist2 += mapped_max_t * mapped_max_t * alpha * T;
				// median depth
				
				C[DEPTH_OFFSET] += root * alpha * T;
			}
			
			if (return_normal){ // and curvature
				float3 point_normal_unnormalized = {2 * p.x * rscale_sign_j.x, 2 * p.y * rscale_sign_j.y, -rscale_sign_j.z};
				float cos_ray_normal = point_normal_unnormalized.x*cam_ray_local.x + point_normal_unnormalized.y*cam_ray_local.y + point_normal_unnormalized.z*cam_ray_local.z;
				float sign_normal = copysign(1.0f, cos_ray_normal);
	
				// Ensure that the normal is oriented outward.
				if (sign_normal > 0){ 
					point_normal_unnormalized.x *= -1.0f;
					point_normal_unnormalized.y *= -1.0f;
					point_normal_unnormalized.z *= -1.0f;
				}
				
				float length = sqrt(point_normal_unnormalized.x * point_normal_unnormalized.x + point_normal_unnormalized.y * point_normal_unnormalized.y + point_normal_unnormalized.z * point_normal_unnormalized.z + 1e-7);
				float3 point_normal = { point_normal_unnormalized.x / length, point_normal_unnormalized.y / length, point_normal_unnormalized.z / length };
				
	
				// transform to view space
				float3 transformed_normal = {
					view2gaussian_j[0] * point_normal.x + view2gaussian_j[1] * point_normal.y + view2gaussian_j[2] * point_normal.z,
					view2gaussian_j[4] * point_normal.x + view2gaussian_j[5] * point_normal.y + view2gaussian_j[6] * point_normal.z,
					view2gaussian_j[8] * point_normal.x + view2gaussian_j[9] * point_normal.y + view2gaussian_j[10] * point_normal.z,
				};
				
				const float normal[3] = { transformed_normal.x, transformed_normal.y, transformed_normal.z};
				// normal
				for (int ch = 0; ch < CHANNELS; ch++)
					C[NORMAL_OFFSET + ch] += normal[ch] * alpha * T;
				// compute gaussian curvature, see the supplementary material of QGS.
				float coeff_1 = scale_j.z * rscale_sign_j.x;
				float coeff_2 = scale_j.z * rscale_sign_j.y;
				float a2u2 = coeff_1 * coeff_1 * p.x * p.x;
				float b2v2 = coeff_2 * coeff_2 * p.y * p.y;
				float den = 1 + 4 * (a2u2 + b2v2);
				float curvature = 4 * coeff_1 * coeff_2 * __frcp_rn(den * den);
				C[CURVATURE_OFFSET] += alpha * T * curvature;
				float error_curv = curvature * curvature * A + curv2 - 2 * curvature * curv1;
				C[CURV_DISTORTION_OFFSET] += error_curv * alpha * T;

				curv1 += curvature * alpha * T;
				curv2 += curvature * curvature * alpha * T;
		
			}
			if (T > 0.5){
				C[MIDDEPTH_OFFSET] = root;
				median_weight = alpha * T;
				median_contributor = contributor;
			}
			
			
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			
			// alpha
			C[ALPHA_OFFSET] += alpha * T;

			T = test_T;

			// Cumulative gradients should take into account whether the current primitive contributed to the rendering.
			// if (T >= 0.5)
			atomicAdd(&(n_touched[collected_id[j]]), 1);
			// Keep track of last range entry to update this pixel.
			// But for per-pixel resorting, recording the last contributor is unnecessary.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// add the background 
		const float distortion_before_normalized = C[DISTORTION_OFFSET];
		// normalize
		// distortion /= (1 - T) * (1 - T) + 1e-7;

		final_T[pix_id] = T; // storing A, D1, D2, dist_unnormal
		if (return_depth){
			final_T[pix_id + H * W] = dist1;
			final_T[pix_id + 2 * H * W] = dist2;
			final_T[pix_id + 3 * H * W] = distortion_before_normalized;
			out_color[DEPTH_OFFSET * H * W + pix_id] = C[DEPTH_OFFSET];
			out_color[DISTORTION_OFFSET * H * W + pix_id] = C[DISTORTION_OFFSET];
		}
		if (return_normal){
			final_T[pix_id + 4 * H * W] = curv1;
			final_T[pix_id + 5 * H * W] = curv2;
			out_color[CURVATURE_OFFSET * H * W + pix_id] = C[CURVATURE_OFFSET];
			out_color[CURV_DISTORTION_OFFSET * H * W + pix_id] = C[CURV_DISTORTION_OFFSET];
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[(NORMAL_OFFSET + ch) * H * W + pix_id] = C[NORMAL_OFFSET+ch];
		}

		n_contrib[pix_id] = last_contributor;
		n_contrib[pix_id + H * W] = median_contributor;
		out_color[MIDDEPTH_OFFSET * H * W + pix_id] = C[MIDDEPTH_OFFSET];

		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_color[ALPHA_OFFSET * H * W + pix_id] = C[ALPHA_OFFSET];
		out_color[MEDIAN_WEIGHT_OFFSET * H * W + pix_id] = C[MEDIAN_WEIGHT_OFFSET];
	}
}



void FORWARD::render(
	const int P,
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const float principal_x, const float principal_y,
	const float sigma,
	const float* colors,
	const float* view2gaussian,
	const float* viewmatrix,
	const float3* means3D,
	const float3* scales,
	const float* depths,
	const float4* rscales_opacity,
	const int3* scales_sign,
	const bool return_depth,
	const bool return_normal,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int* n_touched)
{
#if PIXEL_RESORTING
	renderkBufferCUDA<NUM_CHANNELS> << <grid, block >> > (
		P,
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		principal_x, principal_y,
		sigma,
		colors,
		view2gaussian,
		viewmatrix,
		means3D,
		scales,
		depths,
		rscales_opacity,
		scales_sign,
		return_depth,
		return_normal,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		n_touched);
#else
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		P,
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		principal_x, principal_y,
		sigma,
		colors,
		view2gaussian,
		viewmatrix,
		means3D,
		scales,
		depths,
		rscales_opacity,
		scales_sign,
		return_depth,
		return_normal,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		n_touched);
#endif
}

void FORWARD::preprocess(int P, int D, int M,
	float* aabb,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const float sigma,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* colors_precomp,
	const float* view2gaussian_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float principal_x, float principal_y,
	const float kernel_size,
	int* radii,
	float2* rects,
	// uint2* rects,
	float2* means2D,
	float* depths,
	float* view2gaussians,
	float* rgb,
	float4* rscales_opacity,
	int3* scales_sign,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		aabb,
		means3D,
		scales,
		scale_modifier,
		sigma,
		rotations,
		opacities,
		shs,
		clamped,
		colors_precomp,
		view2gaussian_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		focal_x, focal_y,
		principal_x, principal_y,
		kernel_size,
		radii,
		rects,
		means2D,
		depths,
		view2gaussians,
		rgb,
		rscales_opacity,
		scales_sign,
		grid,
		tiles_touched,
		prefiltered
		);
}


void FORWARD::duplicate(
	int P,
	int W, int H,
	const float focal_x, const float focal_y,
	const float principal_x, const float principal_y,
	const float sigma,
	const float2* means2D,
	const float* depths,
	const float4* rscales_opacity,
	const float3* scales,
	const int3* scales_sign,
	const float* view2gaussians,
	const uint32_t* offsets,
	const int* radii,
	const float2* rects,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	dim3 grid)
{
	duplicateWithKeys_extended<false, true> << <(P + 255) / 256, 256 >> >(
		P, W, H, focal_x, focal_y, principal_x, principal_y, sigma,
		means2D,
		depths,
		rscales_opacity,
		scales,
		scales_sign,
		view2gaussians,
		offsets,
		radii,
		rects,
		gaussian_keys_unsorted,
		gaussian_values_unsorted,
		grid
	);

}
