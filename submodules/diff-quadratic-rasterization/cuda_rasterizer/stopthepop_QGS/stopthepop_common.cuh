/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

 #pragma once

 #include "../auxiliary.h"
 
 #include <cooperative_groups.h>
 namespace cg = cooperative_groups;

__device__ __inline__ uint64_t constructSortKey(uint32_t tile_id, float depth)
{
    uint64_t key = tile_id;
    key <<= 32;
    key |= *((uint32_t*)&depth);
    return key;
}

// Given a ray and a Gaussian primitive, compute the intersection depth.
__device__ __inline__ bool getIntersectPoint(
    const int W, const int H,
    const float fx, const float fy,
    const float cx, const float cy,
    const float sigma,
    const float3 rscale_sign,
    const float3 scale, 
    const glm::vec2 pixel_center,
    const float* view2gaussian,
    float& depth
){
    float2 ray = {(pixel_center.x - cx) / fx, (pixel_center.y - cy) / fy};
    float3 ray_point = { ray.x , ray.y, 1.0 }; 
    
    float3 cam_pos_local = {view2gaussian[12], view2gaussian[13], view2gaussian[14]};
    float3 cam_ray_local = transformPoint4x3_without_t(ray_point, view2gaussian);
 
 
    const double AA = rscale_sign.x * cam_ray_local.x * cam_ray_local.x + 
				      rscale_sign.y * cam_ray_local.y * cam_ray_local.y;
 
    const double BB = 2 * (rscale_sign.x * cam_pos_local.x * cam_ray_local.x + rscale_sign.y * cam_pos_local.y * cam_ray_local.y) - 
				      rscale_sign.z * cam_ray_local.z;
 
    const double CC = rscale_sign.x * cam_pos_local.x * cam_pos_local.x + 
				      rscale_sign.y * cam_pos_local.y * cam_pos_local.y - 
				      rscale_sign.z * cam_pos_local.z;
				      float discriminant = BB*BB - 4*AA*CC;
    if(discriminant < 0)
	    return false;
				      
    float r2AA = __frcp_rn(2 * AA);
    float discriminant_sq_r2AA = __fsqrt_rn(discriminant) * r2AA;
	      
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
		p = {cam_pos_local.x + root * cam_ray_local.x, cam_pos_local.y + root * cam_ray_local.y};
	    p_norm_2 = p.x * p.x + p.y * p.y + 1e-7;
	    p_norm = __fsqrt_rn(p_norm_2);
 
	    cos2_sin2 = {p.x * p.x / p_norm_2, p.y * p.y / p_norm_2};
 
	    a = GetParabolaA(cos2_sin2, rscale_sign, scale);
	    s = QuadraticCurveGeodesicDistanceOriginal(p_norm, a);
		s_2 = s * s;

	    r0_2 = 1 / (cos2_sin2.x * __frcp_rn(scale.x) * __frcp_rn(scale.x) + cos2_sin2.y * __frcp_rn(scale.y) * __frcp_rn(scale.y));
	    if (s_2 <= r0_2 * sigma * sigma){
		    intersect = true;
		    break;
	    }
    }
    if (!intersect)
	    return false;
 
    depth = root;
    return true;
}

 
template<bool TILE_BASED_CULLING = false, bool LOAD_BALANCING = true>
__global__ void duplicateWithKeys_extended(
    int P, 
    int W, int H,
    const float focal_x, const float focal_y,
    const float principal_x, const float principal_y,
    const float sigma,
    const float2* __restrict__ points_xy,
    const float* __restrict__ depths,
    const float4* __restrict__ rscales_opacity,
    const float3* __restrict__ scales,
    const int3* __restrict__ scales_sign,
    const float* __restrict__ view2gaussians,
    const uint32_t* __restrict__  offsets,
    const int* __restrict__ radii,
    const float2* __restrict__ rects,
    uint64_t* __restrict__ gaussian_keys_unsorted,
    uint32_t* __restrict__ gaussian_values_unsorted,
    dim3 grid)
{	
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WARP_SIZE>(block);

	// Since the projection of the quadratic surface on the image is non-convex, 
	// there is no explicit solution for computing the pixel with the maximum weight on the image,
	// and tile-based culling is not performed.
    constexpr bool EVAL_MAX_CONTRIB_POS = false;
    constexpr bool PER_TILE_DEPTH = true;

#define RETURN_OR_INACTIVE() if constexpr(LOAD_BALANCING) { active = false; } else { return; }
    uint32_t idx = cg::this_grid().thread_rank();
    bool active = true;
    if (idx >= P) {
	    RETURN_OR_INACTIVE();
	    idx = P - 1;
    }

    const int radius = radii[idx];
    if (radius <= 0) {
	    RETURN_OR_INACTIVE();
    }

	// If the thread exceeds the Gaussian index, the Gaussian projection is zero, 
	// and there are no Gaussians to process in the current warp, return.
    if constexpr(LOAD_BALANCING)
	    if (__ballot_sync(WARP_MASK, active) == 0)
		    return;

    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off_init = (idx == 0) ? 0 : offsets[idx - 1];

    const int offset_to_init = offsets[idx];
    const float global_depth_init = depths[idx];

    const float2 xy_init = points_xy[idx];
    const float2 rect_dims_init = rects[idx];

    __shared__ float2 s_xy[BLOCK_SIZE];
    __shared__ float2 s_rect_dims[BLOCK_SIZE];
    s_xy[block.thread_rank()] = xy_init;
    s_rect_dims[block.thread_rank()] = rect_dims_init;

    uint2 rect_min_init, rect_max_init;
    getRect(xy_init, rect_dims_init, rect_min_init, rect_max_init, grid);

    __shared__ float s_view2gaussians[BLOCK_SIZE * 16];
    __shared__ float3 s_scales[BLOCK_SIZE];
    __shared__ float4 s_rscales_opacity[BLOCK_SIZE];
    __shared__ int3 s_scales_sign[BLOCK_SIZE];

    if (PER_TILE_DEPTH)
    {
	    s_rscales_opacity[block.thread_rank()] = rscales_opacity[idx];
	    s_scales[block.thread_rank()] = scales[idx];
	    s_scales_sign[block.thread_rank()] = scales_sign[idx];
	    for (int ii = 0; ii < 16; ii++)
		    s_view2gaussians[16 * block.thread_rank() + ii] = view2gaussians[idx * 16 + ii];
    }

    constexpr uint32_t SEQUENTIAL_TILE_THRESH = 32U; // all tiles above this threshold will be computed cooperatively
    const uint32_t rect_width_init = (rect_max_init.x - rect_min_init.x);
    const uint32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * rect_width_init;

    // Generate no key/value pair for invisible Gaussians
    if (tile_count_init == 0)	{
	    RETURN_OR_INACTIVE();
    }
    auto tile_function = [&](const int W, const int H,
						     const float fx, const float fy,
						     const float cx, const float cy,
						     const float sigma,
						     float2 xy,
						     int x, int y,// tile ID
						     const float3 rscale_sign, 
						     const float3 scale, 
						     const float* view2gaussian, 
						     const float global_depth,
						     float& depth)  
	    {
		    const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
		    const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1); // 像素坐标

		    glm::vec2 max_pos;
		    if constexpr (PER_TILE_DEPTH) 
		    {	
			    glm::vec2 target_pos = {max(min(xy.x, tile_max.x), tile_min.x), max(min(xy.y, tile_max.y), tile_min.y)};

				// Or select the tile's center pixel as the target_pos.
			    // const glm::vec2 tile_center = (tile_min + tile_max) * 0.5f;
			    // glm::vec2 target_pos = tile_center;

			    bool intersect = getIntersectPoint(
				    W, H, fx, fy, cx, cy, sigma, 
				    rscale_sign, scale, target_pos, view2gaussian, depth); // Compute the intersection point of the quadratic surface.
			    if (intersect)
				    depth = max(0.0f, depth);
			    else // If there is no intersection, sort by the Gaussian centroid.
				    depth = global_depth;
		    }
		    else
		    {
			    depth = global_depth;
		    }

			// Since the quadratic surface is non-convex, tile-based culling is not performed.
		    // return (!TILE_BASED_CULLING) || max_opac_factor <= opacity_factor_threshold;
		    return true; 
	    };

    if (active)
    {
	    const float3 rscale_sign_init = {
		    s_rscales_opacity[block.thread_rank()].x * (float)s_scales_sign[block.thread_rank()].x, 
		    s_rscales_opacity[block.thread_rank()].y * (float)s_scales_sign[block.thread_rank()].y, 
		    s_rscales_opacity[block.thread_rank()].z};
	    const float3 scale_init = {
		    s_scales[block.thread_rank()].x, 
		    s_scales[block.thread_rank()].y, 
		    s_scales[block.thread_rank()].z};

	    float view2gaussian_init[16];
	    for (int ii = 0; ii < 16; ii++)
		    view2gaussian_init[ii] = s_view2gaussians[16 * block.thread_rank() + ii];

	    for (uint32_t tile_idx = 0; tile_idx < tile_count_init && (!LOAD_BALANCING || tile_idx < SEQUENTIAL_TILE_THRESH); tile_idx++)
	    {
		    const int y = (tile_idx / rect_width_init) + rect_min_init.y;
		    const int x = (tile_idx % rect_width_init) + rect_min_init.x;

		    float depth;
		    bool write_tile = tile_function(
				    W, H, focal_x, focal_y, principal_x, principal_y, sigma,
				    xy_init, x, y, rscale_sign_init, scale_init, view2gaussian_init, global_depth_init, depth);
		    if (write_tile)
		    {
			    if (off_init < offset_to_init)
			    {
				    const uint32_t tile_id = y * grid.x + x;
				    gaussian_values_unsorted[off_init] = idx;
				    gaussian_keys_unsorted[off_init] = constructSortKey(tile_id, depth);
			    }
			    else
			    {
#ifdef DUPLICATE_OPT_DEBUG
				    printf("Error (sequential): Too little memory reserved in preprocess: off=%d off_to=%d idx=%d\n", off_init, offset_to_init, idx);
#endif
			    }
			    off_init++;
		    }
	    }
    }

#undef RETURN_OR_INACTIVE

    if (!LOAD_BALANCING) // Coordinate to handle the unprocessed tasks of other threads within the same warp.
	    return;

    const uint32_t idx_init = idx; // Current thread idx.
    const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
    const uint32_t warp_idx = cg::this_thread_block().thread_rank() / WARP_SIZE;
    unsigned int lane_mask_allprev_excl = 0xFFFFFFFFU >> (WARP_SIZE - lane_idx);

    const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH; // Determine whether additional idle threads are needed for computation.
    const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);
    if (remaining_threads == 0)
	    return;
 
    uint32_t n_remaining_threads = __popc(remaining_threads); // The number of threads required for collaborative computation.
    for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++) 
    {
	    int i = __fns(remaining_threads, 0, n+1); // find lane index of next remaining thread

	    uint32_t idx_coop = __shfl_sync(WARP_MASK, idx_init, i); 
	    uint32_t off_coop = __shfl_sync(WARP_MASK, off_init, i); 

	    const uint32_t offset_to = __shfl_sync(WARP_MASK, offset_to_init, i);
	    const float global_depth = __shfl_sync(WARP_MASK, global_depth_init, i);

	    const float2 xy = s_xy[warp.meta_group_rank() * WARP_SIZE + i];
	    const float2 rect_dims = s_rect_dims[warp.meta_group_rank() * WARP_SIZE + i]; 
	    const float3 rscale_sign = {
		    s_rscales_opacity[warp.meta_group_rank() * WARP_SIZE + i].x * (float)s_scales_sign[warp.meta_group_rank() * WARP_SIZE + i].x, 
		    s_rscales_opacity[warp.meta_group_rank() * WARP_SIZE + i].y * (float)s_scales_sign[warp.meta_group_rank() * WARP_SIZE + i].y, 
		    s_rscales_opacity[warp.meta_group_rank() * WARP_SIZE + i].z};
	    const float3 scale = {
		    s_scales[warp.meta_group_rank() * WARP_SIZE + i].x, 
		    s_scales[warp.meta_group_rank() * WARP_SIZE + i].y, 
		    s_scales[warp.meta_group_rank() * WARP_SIZE + i].z};
	    float view2gaussian[16];
	    for (int ii = 0; ii < 16; ii++)
		    view2gaussian[ii] = s_view2gaussians[16 * (warp.meta_group_rank() * WARP_SIZE + i) + ii];

	    uint2 rect_min, rect_max;
	    getRect(xy, rect_dims, rect_min, rect_max, grid);

	    const uint32_t rect_width = (rect_max.x - rect_min.x);
	    const uint32_t tile_count = (rect_max.y - rect_min.y) * rect_width;
	    const uint32_t remaining_tile_count = tile_count - SEQUENTIAL_TILE_THRESH;
	    const int32_t n_iterations = (remaining_tile_count + WARP_SIZE - 1) / WARP_SIZE;
	    for (int it = 0; it < n_iterations; it++)
	    {
		    int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH; // it*32 + local_warp_idx + 32
		    int active_curr_it = tile_idx < tile_count;
 
		    int y = (tile_idx / rect_width) + rect_min.y;
		    int x = (tile_idx % rect_width) + rect_min.x;

		    float depth;
		    bool write_tile = tile_function(
			    W, H, focal_x, focal_y, principal_x, principal_y, sigma,
			    xy, x, y, rscale_sign, scale, view2gaussian, global_depth, depth
		    );

		    const uint32_t write = active_curr_it && write_tile;

		    uint32_t n_writes, write_offset;
		    if constexpr (!TILE_BASED_CULLING)
		    {
			    n_writes = WARP_SIZE;
			    write_offset = off_coop + lane_idx;
		    }
		    else
		    {
			    const uint32_t write_ballot = __ballot_sync(WARP_MASK, write);
			    n_writes = __popc(write_ballot);
 
			    const uint32_t write_offset_it = __popc(write_ballot & lane_mask_allprev_excl);
			    write_offset = off_coop + write_offset_it;
		    }

		    if (write)
		    {
			    if (write_offset < offset_to)
			    {
				    const uint32_t tile_id = y * grid.x + x;
				    gaussian_values_unsorted[write_offset] = idx_coop;
				    gaussian_keys_unsorted[write_offset] = constructSortKey(tile_id, depth);
			    }
 #ifdef DUPLICATE_OPT_DEBUG
			    else
			    {
				    printf("Error (parallel): Too little memory reserved in preprocess: off=%d off_to=%d idx=%d tile_count=%d it=%d | x=%d y=%d rect=(%d %d - %d %d)\n", 
							write_offset, offset_to, idx_coop, tile_count, it, x, y, rect_min.x, rect_min.y, rect_max.x, rect_max.y);
			    }
 #endif
		    }
		    off_coop += n_writes;
	    }

	    __syncwarp();
    }
 }