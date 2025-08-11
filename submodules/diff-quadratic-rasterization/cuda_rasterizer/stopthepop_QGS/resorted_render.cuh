
// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.

#include "../auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


__device__ __inline__ void updateDistortionMap(
    const float depth,
    const float alpha,
    const float T,
    const float curvature,
    float& dist1,
    float& dist2,
    float& curv1,
    float& curv2,
    float* all_map
) {
    float A = 1 - T;
	float w = alpha * T;
    float mapped_max_t = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
    float error = mapped_max_t * mapped_max_t * A + dist2 - 2 * mapped_max_t * dist1;
	all_map[DISTORTION_OFFSET] += error * w;

	dist1 += mapped_max_t * w;
	dist2 += mapped_max_t * mapped_max_t * w;

    float error_curv = curvature * curvature * A + curv2 - 2 * curvature * curv1;
    all_map[CURV_DISTORTION_OFFSET] += error_curv * w;

    curv1 += curvature * w;
    curv2 += curvature * curvature * w;
}

template <uint32_t CHANNELS>
__device__ __inline__ void updateMap(
    const float depth,
    const float alpha,
    const float T,
    const float* normal,
    const float curvature,
    float* all_map
) {
	float w = alpha * T;
    for (int ch = 0; ch < CHANNELS; ch++)
		all_map[NORMAL_OFFSET + ch] += normal[ch] * w;
    all_map[DEPTH_OFFSET] += depth * w;
	all_map[ALPHA_OFFSET] += w;
	all_map[CURVATURE_OFFSET] += w * curvature;
    if (T > 0.5)
        all_map[MIDDEPTH_OFFSET] = depth;
}




template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderkBufferCUDA(
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
	float* __restrict__ out_colors,
	int* __restrict__ n_touched)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f}; // TODO plus 0.5

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// create the ray
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


	// Used for buffering.
    float sort_depths[BUFFER_LENGTH];
	float sort_alphas[BUFFER_LENGTH];
    float sort_normals[BUFFER_LENGTH * 3];
    float sort_curvs[BUFFER_LENGTH];
	int sort_ids[BUFFER_LENGTH];
	int sort_num = 0;
    for (int i = 0; i < BUFFER_LENGTH; ++i)
	{
		sort_depths[i] = FLT_MAX;
		// just to suppress warnings:
		sort_alphas[i] = 0;
        sort_curvs[i] = 0;
        for (int ch = 0; ch < CHANNELS; ch++)
            sort_normals[i * CHANNELS + ch] = 0;
		sort_ids[i] = -1;
	}

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t median_contributor = -1;
	float C[OUTPUT_CHANNELS] = {0};

	float dist1 = {0};
	float dist2 = {0};
	float curv1 = {0};
	float curv2 = {0};
	float median_weight = {0};

    auto blend_one = [&](){
        if (sort_num == 0)
            return;
        --sort_num;
        float test_T = T * (1 - sort_alphas[0]);

        if (test_T < 0.0001f) {
            done = true;
            return;
        }
        updateDistortionMap(sort_depths[0], sort_alphas[0], T, sort_curvs[0], dist1, dist2, curv1, curv2, C);
        updateMap<CHANNELS>(sort_depths[0], sort_alphas[0], T, sort_normals, sort_curvs[0], C);
        for (int ch = 0; ch < CHANNELS; ch++)
			C[ch] += features[sort_ids[0] * CHANNELS + ch] * sort_alphas[0] * T;

        if (T > 0.5){
            median_contributor = contributor;
			C[MIDDEPTH_OFFSET] = sort_depths[0];
			
		}

        T = test_T;
		atomicAdd(&(n_touched[sort_ids[0]]), 1);

        for (int i = 1; i < BUFFER_LENGTH; ++i)
		{
			sort_depths[i - 1] = sort_depths[i];
			sort_alphas[i - 1] = sort_alphas[i];
			sort_ids[i - 1] = sort_ids[i];
			if (return_normal){
				for (int ch = 0; ch < CHANNELS; ch++)
                	sort_normals[(i - 1) * CHANNELS + ch] = sort_normals[i * CHANNELS + ch];
            	sort_curvs[i - 1] = sort_curvs[i];
			}
            
		}
		sort_depths[BUFFER_LENGTH - 1] = FLT_MAX;
    };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int all_done = __syncthreads_and(done);
		if (all_done)
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

			// When the buffer is full, immediately retrieve the Gaussian rendering that is the closest.
            if (sort_num == BUFFER_LENGTH)
                blend_one();
			
            if (done == true)
                break;

            contributor++;
            
			int global_id = collected_id[j];
            

			const float* view2gaussian_j = collected_view2gaussian + j * 16;
			const float3 scale_sign_j = {(float)collected_scales_sign[j].x, (float)collected_scales_sign[j].y, (float)collected_scales_sign[j].z};
			const float3 scale_j = collected_scales[j];
			float4 rscale_o_j = collected_rscales_opacity[j];
			const float3 rscale_sign_j = {rscale_o_j.x * scale_sign_j.x, rscale_o_j.y * scale_sign_j.y, rscale_o_j.z};
			
			
			// transform camera center and ray to gaussian's local coordinate system
			// current center is zero
			float3 cam_pos_local = {view2gaussian_j[12], view2gaussian_j[13], view2gaussian_j[14]};
			float3 cam_ray_local = transformPoint4x3_without_t(ray_point, view2gaussian_j);

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
			for(int i = -1; i < 2; i += 2){

#if	QUADRATIC_APPROXIMATION
				if (abs(AA) < 1e-6) // approximation of the intersection equation, see the supplementary material.
				root = __fdividef(-CC, BB);
				else
#endif				
				{
					sign = (float)i * (float)AA_sign;
					root = -BB * r2AA + sign * discriminant_sq_r2AA;
				}
				p = {
					__fmaf_rn(root, cam_ray_local.x, cam_pos_local.x),
					__fmaf_rn(root, cam_ray_local.y, cam_pos_local.y)
				};
				p_norm_2 = p.x * p.x + p.y * p.y + 1e-7;
				p_norm = __fsqrt_rn(p_norm_2);
				cos2_sin2 = {p.x * p.x / p_norm_2, p.y * p.y / p_norm_2};
				a = GetParabolaA(cos2_sin2, rscale_sign_j, scale_j);
				s = QuadraticCurveGeodesicDistanceOriginal(p_norm, a);
				s_2 = s * s;
				r0_2 = __fdividef(1.0f, (cos2_sin2.x * rscale_o_j.x + cos2_sin2.y * rscale_o_j.y));
				// if the Gaussian weight at the intersection is too small, skip it.
				if (s_2 <= r0_2 * sigma * sigma){
					intersect = true;
					break;
				}
			}

			if (!intersect)
                continue;
			
			float depth = root;

            if (depth < 0.0f)
                continue;

			float normal[CHANNELS] = {0};
			float curvature = 0.0f;
			if (return_normal){
				float3 point_normal_unnormalized = {2 * p.x * rscale_sign_j.x, 2 * p.y * rscale_sign_j.y, -rscale_sign_j.z};
				float cos_ray_normal = point_normal_unnormalized.x * cam_ray_local.x + 
									   point_normal_unnormalized.y * cam_ray_local.y + 
									   point_normal_unnormalized.z * cam_ray_local.z;
				float sign_normal = copysign(1.0f, cos_ray_normal);
				if (sign_normal > 0){
					point_normal_unnormalized.x = -point_normal_unnormalized.x;
					point_normal_unnormalized.y = -point_normal_unnormalized.y;
					point_normal_unnormalized.z = -point_normal_unnormalized.z;
				}
				float length = sqrt(point_normal_unnormalized.x * point_normal_unnormalized.x + point_normal_unnormalized.y * point_normal_unnormalized.y + point_normal_unnormalized.z * point_normal_unnormalized.z + 1e-6);
				float3 point_normal = { point_normal_unnormalized.x / length, point_normal_unnormalized.y / length, point_normal_unnormalized.z / length };
				
				// transform to view space
				float3 transformed_normal = {
					view2gaussian_j[0] * point_normal.x + view2gaussian_j[1] * point_normal.y + view2gaussian_j[2] * point_normal.z,
					view2gaussian_j[4] * point_normal.x + view2gaussian_j[5] * point_normal.y + view2gaussian_j[6] * point_normal.z,
					view2gaussian_j[8] * point_normal.x + view2gaussian_j[9] * point_normal.y + view2gaussian_j[10] * point_normal.z,
				};
				// float normal[CHANNELS] = { transformed_normal.x, transformed_normal.y, transformed_normal.z};
				normal[0] = transformed_normal.x;
				normal[1] = transformed_normal.y;
				normal[2] = transformed_normal.z;
				
				// Gaussian curvature
				float coeff_1 = scale_j.z * rscale_sign_j.x;
				float coeff_2 = scale_j.z * rscale_sign_j.y;
				float a2u2 = coeff_1 * coeff_1 * p.x * p.x;
				float b2v2 = coeff_2 * coeff_2 * p.y * p.y;
				float den = 1 + 4 * (a2u2 + b2v2);
				curvature = 4 * coeff_1 * coeff_2 * __frcp_rn(den * den);
			}
			
			float power = __fdividef(-s_2, (2.0f * r0_2));
			// float power = -s_2 / (2 * r0_2);
			if (power > 0.0f)
				continue;
            

			float alpha = min(0.99f, rscale_o_j.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;

			int id = collected_id[j];
#pragma unroll
            for (int s = 0; s < BUFFER_LENGTH; s++){
                
                if (depth < sort_depths[s]){
					swap_T(depth, sort_depths[s]);
					swap_T(alpha, sort_alphas[s]);
					swap_T(id, sort_ids[s]);
					if (return_normal){
						for (int ch = 0; ch < CHANNELS; ch++)
							swap_T(normal[ch], sort_normals[s*CHANNELS + ch]);
						swap_T(curvature, sort_curvs[s]);
					}
					
                }
            }
			++sort_num;
		}
	}
	if (!done) {
		while (sort_num > 0)
			blend_one();
	}

	if (inside)
	{
		final_T[pix_id] = T;

		if (return_depth){
			final_T[pix_id + H * W] = dist1;
			final_T[pix_id + 2 * H * W] = dist2;
			final_T[pix_id + 3 * H * W] = C[DISTORTION_OFFSET];
			out_colors[DEPTH_OFFSET * H * W + pix_id] = C[DEPTH_OFFSET];
			out_colors[DISTORTION_OFFSET * H * W + pix_id] = C[DISTORTION_OFFSET];
			out_colors[MIDDEPTH_OFFSET * H * W + pix_id] = C[MIDDEPTH_OFFSET];
			out_colors[MEDIAN_WEIGHT_OFFSET * H * W + pix_id] = C[MEDIAN_WEIGHT_OFFSET];
		}

		if (return_normal){
			for (int ch = 0; ch < CHANNELS; ch++)
				out_colors[(NORMAL_OFFSET + ch) * H * W + pix_id] = C[NORMAL_OFFSET+ch];
			out_colors[CURVATURE_OFFSET * H * W + pix_id] = C[CURVATURE_OFFSET];
			out_colors[CURV_DISTORTION_OFFSET * H * W + pix_id] = C[CURV_DISTORTION_OFFSET];
		}

		final_T[pix_id + 4 * H * W] = curv1;
		final_T[pix_id + 5 * H * W] = curv2;

		n_contrib[pix_id] = last_contributor;
		n_contrib[pix_id + H * W] = median_contributor;

		for (int ch = 0; ch < CHANNELS; ch++)
			out_colors[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_colors[ALPHA_OFFSET * H * W + pix_id] = C[ALPHA_OFFSET];
	}
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderkBufferBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	float principal_x, float principal_y,
	const float sigma, 
	const bool stop_z_gradient,
	const float* __restrict__ bg_color,
	const float4* __restrict__ rscales_opacity,
	const int3* __restrict__ scales_sign,
	const float* __restrict__ colors,
	const float* __restrict__ view2gaussian,
	const float* viewmatrix,
	const float3* __restrict__ means3D,
	const float3* __restrict__ scales,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ out_colors,
	const float* __restrict__ dL_dpixels,
	const bool return_depth,
	const bool return_normal,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dscales,
	float* __restrict__ dL_dview2gaussian)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f };
 
	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
 
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	
	bool done = !inside;
	int toDo = range.y - range.x;

	float2 ray = { (pixf.x - principal_x) / focal_x, (pixf.y - principal_y) / focal_y };
	float3 ray_point = { ray.x , ray.y, 1.0 };
 
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_rscales_opacity[BLOCK_SIZE];
	__shared__ float3 collected_scales[BLOCK_SIZE];
	__shared__ int3 collected_scales_sign[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	
	// Note that we still use forward traversal during backpropagation.
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = 1.0f; // Thus, the transmittance is initially set to 1.
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_C = inside ? final_Ts[pix_id + 4 * H * W] : 0;
	const float final_C2 = inside ? final_Ts[pix_id + 5 * H * W] : 0;
	const float final_A = 1 - T_final;
	float acc_colors[C] = { 0, 0, 0 };
	float acc_depths = { 0 };
	float acc_normals[C] = { 0, 0, 0 };
	float acc_curvs = { 0 };
	float acc_alphas = { 0 };
	
	float sort_depths[BUFFER_LENGTH];
	float sort_Gs[BUFFER_LENGTH];
	float2 sort_ps[BUFFER_LENGTH];
	float sort_r0_2s[BUFFER_LENGTH];
	float sort_as[BUFFER_LENGTH];
	float sort_ss[BUFFER_LENGTH];
	float sort_signs[BUFFER_LENGTH];
	float4 sort_scales_os[BUFFER_LENGTH];

	int sort_ids[BUFFER_LENGTH];
	int sort_num = 0;
    for (int i = 0; i < BUFFER_LENGTH; ++i)
	{
		sort_depths[i] = FLT_MAX;
		// just to suppress warnings:
		sort_Gs[i] = 0;
		sort_ps[i] = {0.0f, 0.0f};
		sort_r0_2s[i] = 0.0f;
		sort_as[i] = 0.0f;
		sort_ss[i] = 0.0f;
		sort_signs[i] = 0.0f;
		sort_scales_os[i] = {0.0f, 0.0f, 0.0f, 0.0f};
		
		sort_ids[i] = -1;
	}


	float last_dL_dT = 0;
	
	// We start from the front.
	uint32_t contributor = 0;
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float accum_rec[C] = { 0 };
	float dL_dpixel[C]; // RGB
	float dL_dnormal2D[3];
	float dL_daccum = 0.0f;
	float dL_dreg = 0.0f;
	float dL_ddepth = 0.0f;
	float dL_dmedian_depth = 0.0f;
	float dL_dmax_dweight = 0.0f;
	float dL_dcurv = 0.0f;
	float dL_dreg_curv = 0.0f;

	float final_color[C];
	float final_normal[C];
	float final_depth;
	float final_curv;
	float final_alpha;
	
	if (inside){
		for (int i = 0; i < C; i++){
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			final_color[i] = out_colors[i * H * W + pix_id] - T_final * bg_color[i];

			dL_dnormal2D[i] = dL_dpixels[(NORMAL_OFFSET + i) * H * W + pix_id];
			final_normal[i] = out_colors[(NORMAL_OFFSET + i) * H * W + pix_id];
		}

		final_depth = out_colors[DEPTH_OFFSET * H * W + pix_id];
		final_curv = out_colors[CURVATURE_OFFSET * H * W + pix_id];
		final_alpha = out_colors[ALPHA_OFFSET * H * W + pix_id];
		dL_ddepth = dL_dpixels[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_dpixels[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_dpixels[DISTORTION_OFFSET * H * W + pix_id];
		dL_dmedian_depth = dL_dpixels[MIDDEPTH_OFFSET * H * W + pix_id];
		dL_dmax_dweight = dL_dpixels[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
		dL_dcurv = dL_dpixels[CURVATURE_OFFSET * H * W + pix_id];
		dL_dreg_curv = dL_dpixels[CURV_DISTORTION_OFFSET * H * W + pix_id];
	}

	auto blend_one = [&]() {
		if (sort_num == 0)
			return;
		--sort_num;
		int global_id = sort_ids[0]; 
		float G = sort_Gs[0];
		float2 p = sort_ps[0];
		float r0_2 = sort_r0_2s[0];
		float a = sort_as[0];
		float s = sort_ss[0];
		float sign = sort_signs[0];
		float depth = sort_depths[0];

		// first-order
		const float4 scale_o_blend = sort_scales_os[0];
		const float3 scale_sign_j = {(float)copysign(1.0f, scale_o_blend.x), 
									 (float)copysign(1.0f, scale_o_blend.y), 
									 (float)copysign(1.0f, scale_o_blend.z)};

		// const float3 scale_blend = scales[global_id];
		// const float4 rscale_o_blend = rscales_opacity[global_id];
		const float4 rscale_o_blend = {__frcp_rn(scale_o_blend.x * scale_o_blend.x), 
									   __frcp_rn(scale_o_blend.y * scale_o_blend.y), 
									   __frcp_rn(scale_o_blend.z),
									   scale_o_blend.w};

		const float3 rscale_sign_blend = {rscale_o_blend.x * scale_sign_j.x, rscale_o_blend.y * scale_sign_j.y, rscale_o_blend.z}; 
		
		float view2gaussian_blend[16];

		
		view2gaussian_blend[0] = view2gaussian[global_id * 16 + 0];
		view2gaussian_blend[1] = view2gaussian[global_id * 16 + 1];
		view2gaussian_blend[2] = view2gaussian[global_id * 16 + 2];
		view2gaussian_blend[4] = view2gaussian[global_id * 16 + 4];
		view2gaussian_blend[5] = view2gaussian[global_id * 16 + 5];
		view2gaussian_blend[6] = view2gaussian[global_id * 16 + 6];
		view2gaussian_blend[8] = view2gaussian[global_id * 16 + 8];
		view2gaussian_blend[9] = view2gaussian[global_id * 16 + 9];
		view2gaussian_blend[10] = view2gaussian[global_id * 16 + 10];
		view2gaussian_blend[12] = view2gaussian[global_id * 16 + 12];
		view2gaussian_blend[13] = view2gaussian[global_id * 16 + 13];
		view2gaussian_blend[14] = view2gaussian[global_id * 16 + 14];
		// second-order 
		float3 cam_pos_local = {view2gaussian_blend[12], view2gaussian_blend[13], view2gaussian_blend[14]};	
		float3 cam_ray_local = transformPoint4x3_without_t(ray_point, view2gaussian_blend);
		
		float px_2 = p.x * p.x;
		float py_2 = p.y * p.y;
		float p_norm_2 = px_2 + py_2 + 1e-7;
		float p_norm = __fsqrt_rn(p_norm_2);
		float2 cos2_sin2 = {__fdividef(px_2, p_norm_2), __fdividef(py_2, p_norm_2)};
		const float alpha = min(0.99f, rscale_o_blend.w * G);
		float test_T = T * (1 - alpha);
		if(test_T  < 0.0001f){
			done = true;
			return;
		}
		
		const float weight = alpha * T;
		float dL_dalpha = 0.0f;
		for (int ch = 0; ch < C; ch++){
			// The rewritten version of the forward traversal gradient, provided in the QGS supplementary material.
			const float c = colors[global_id * C + ch];
			acc_colors[ch] += c * weight;
			
			float accum_rec_ch = __fdividef((final_color[ch] - acc_colors[ch]), test_T);
			const float dL_dchannel = dL_dpixel[ch];

			dL_dalpha += (c - accum_rec_ch) * dL_dchannel;
			atomicAdd(&(dL_dcolors[global_id * C + ch]), weight * dL_dchannel);
		}
		float dL_dweight = 0.0f;
#if !DETACH_CURVATURE
		float coeff_1 = scale_o_blend.z * rscale_sign_blend.x;
		float coeff_2 = scale_o_blend.z * rscale_sign_blend.y;

		float a2u2 = coeff_1 * coeff_1 * px_2;
		float b2v2 = coeff_2 * coeff_2 * py_2;
		float den = 1 + 4 * (a2u2 + b2v2);
		// gaussian curvature
		float curvature = 4 * coeff_1 * coeff_2 * __frcp_rn(den * den);

		// curvature distortion to weight and curvature
		dL_dweight += (final_C2 + curvature * curvature * final_A - 2 * curvature * final_C) * dL_dreg_curv;
		float dL_dcurvature = 2.0f * (T * alpha) * (curvature * final_A - final_C) * dL_dreg_curv;
		const float rden_3 = __frcp_rn(den * den * den);
		const float dL_dcoeff1 = ((curvature / coeff_1) - (64 * a2u2 * coeff_2) * rden_3) * dL_dcurvature;
		const float dL_dcoeff2 = ((curvature / coeff_2) - (64 * b2v2 * coeff_1) * rden_3) * dL_dcurvature;

		// curvature map to alpha 
		acc_curvs += curvature * weight; // i
		float accum_curv_rec = (final_curv - acc_curvs) / test_T;
		dL_dalpha += (curvature - accum_curv_rec) * dL_dcurv;
		
		// curvature map to curvature
		dL_dcurvature += weight * dL_dcurv;
#endif
		float dL_dmax_t = 0.0f;
		if (return_depth){
			// depth distortion to weight and depth
			const float mapped_max_t = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
			float dmax_t_dd = (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth * depth);
			dL_dweight += (final_D2 + mapped_max_t * mapped_max_t * final_A - 2 * mapped_max_t * final_D) * dL_dreg;	

			// distortion to depth
			dL_dmax_t = 2.0f * (T * alpha) * (mapped_max_t * final_A - final_D) * dL_dreg * dmax_t_dd;
			// depth map to alpha
#if DETACH_DIST2WEIGHT
			dL_dalpha += 0.0f;
#else
			dL_dalpha += dL_dweight - last_dL_dT;
#endif
			
			acc_depths += depth * weight;
			float accum_depth_rec = __fdividef((final_depth - acc_depths), test_T);
			// float accum_depth_rec = (final_depth - acc_depths) / test_T;
			dL_dalpha += (depth - accum_depth_rec) * dL_ddepth;
		}
				

		last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;


		// alpha map to alpha
		acc_alphas += weight;
		float accum_alpha_rec = __fdividef((final_alpha - acc_alphas), test_T);
		// float accum_alpha_rec = (final_alpha - acc_alphas) / test_T;
		dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

		float3 dL_point_normal_unnormalized = {0.0f, 0.0f, 0.0f};
		float dL_dnormal_x = 0.0f;
		float dL_dnormal_y = 0.0f;
		float dL_dnormal_z = 0.0f;
		float3 point_normal = {0.0f, 0.0f, 0.0f};
		if (return_normal){
			// normal regularzation
			float3 point_normal_unnormalized = {2 * p.x * rscale_sign_blend.x, 2 * p.y * rscale_sign_blend.y, -rscale_sign_blend.z};
			float cos_ray_normal = point_normal_unnormalized.x * cam_ray_local.x + 
								   point_normal_unnormalized.y * cam_ray_local.y + 
								   point_normal_unnormalized.z * cam_ray_local.z;
			float sign_normal = copysign(1.0f, cos_ray_normal);
			if (sign_normal > 0){
				point_normal_unnormalized.x = -point_normal_unnormalized.x;
				point_normal_unnormalized.y = -point_normal_unnormalized.y;
				point_normal_unnormalized.z = -point_normal_unnormalized.z;
			}

			float sum_sq = __fmaf_rn(point_normal_unnormalized.x, point_normal_unnormalized.x,
						   __fmaf_rn(point_normal_unnormalized.y, point_normal_unnormalized.y,
						   __fmaf_rn(point_normal_unnormalized.z, point_normal_unnormalized.z, 1e-7f)));
			float length = __fsqrt_rn(sum_sq);
			const float rlength = __frcp_rn(length);
						
			point_normal.x = point_normal_unnormalized.x * rlength;
			point_normal.y = point_normal_unnormalized.y * rlength;
			point_normal.z = point_normal_unnormalized.z * rlength;
			

			float nx = __fmaf_rn(view2gaussian_blend[0], point_normal.x,
					   __fmaf_rn(view2gaussian_blend[1], point_normal.y,
								 view2gaussian_blend[2] * point_normal.z));
 			float ny = __fmaf_rn(view2gaussian_blend[4], point_normal.x,
					   __fmaf_rn(view2gaussian_blend[5], point_normal.y,
								 view2gaussian_blend[6] * point_normal.z));
 			float nz = __fmaf_rn(view2gaussian_blend[8], point_normal.x,
				       __fmaf_rn(view2gaussian_blend[9], point_normal.y,
					             view2gaussian_blend[10]* point_normal.z));

 			float normal[3] = {nx, ny, nz};

					
			float dL_dnormal_reg[3] = {0};
			// // Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				// normal map to alpha
				acc_normals[ch] += normal[ch] * weight;
				float accum_normal_rec_ch = __fdividef((final_normal[ch] - acc_normals[ch]), test_T);
				// float accum_normal_rec_ch = (final_normal[ch] - acc_normals[ch]) / test_T;
				dL_dalpha += (normal[ch] - accum_normal_rec_ch) * dL_dnormal2D[ch];

				// normal map to normal
				dL_dnormal_reg[ch] = weight * dL_dnormal2D[ch];
			}
			dL_dnormal_x = dL_dnormal_reg[0];
			dL_dnormal_y = dL_dnormal_reg[1];
			dL_dnormal_z = dL_dnormal_reg[2];

			const float3 dL_dpoint_normal = { 
				view2gaussian_blend[0] * dL_dnormal_x + view2gaussian_blend[4] * dL_dnormal_y + view2gaussian_blend[8] * dL_dnormal_z,
				view2gaussian_blend[1] * dL_dnormal_x + view2gaussian_blend[5] * dL_dnormal_y + view2gaussian_blend[9] * dL_dnormal_z,
				view2gaussian_blend[2] * dL_dnormal_x + view2gaussian_blend[6] * dL_dnormal_y + view2gaussian_blend[10] * dL_dnormal_z
			};

			float dL_dlength = dL_dpoint_normal.x * point_normal_unnormalized.x + dL_dpoint_normal.y * point_normal_unnormalized.y + dL_dpoint_normal.z * point_normal_unnormalized.z;
			dL_dlength *= -1.f / (length * length);
			dL_point_normal_unnormalized.x = (dL_dpoint_normal.x + dL_dlength * point_normal_unnormalized.x) * rlength * (-sign_normal);
			dL_point_normal_unnormalized.y = (dL_dpoint_normal.y + dL_dlength * point_normal_unnormalized.y) * rlength * (-sign_normal);
			dL_point_normal_unnormalized.z = (dL_dpoint_normal.z + dL_dlength * point_normal_unnormalized.z) * rlength * (-sign_normal);

		}
		

		dL_dalpha *= T;

		// Because the rewritten version uses a different recursive form, we do not need to update last_alpha.
		// last_alpha = alpha;
		
		// Account for fact that alpha also influences how much of
		// the background color is added if nothing left to blend
		float bg_dot_dpixel = 0;
		for (int i = 0; i < C; i++)
			bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
		// dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
		dL_dalpha += __fdividef(-T_final, 1.f - alpha) * bg_dot_dpixel;
 
		atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		const float dL_dG = rscale_o_blend.w * dL_dalpha;

		const float dL_ds = dL_dG * (-G * s / r0_2);
		const float dL_dr0_2 = dL_dG * (G * s * s / (2 * r0_2 * r0_2));
		const float u = 2 * a * p_norm;
		const float dL_du = dL_ds * sqrt(u * u + 1) / (2 * a); 

		float dL_da = 2 * dL_du * p_norm - dL_ds * s / a;

		float rcos_s_sin_s_2_2 = r0_2 * r0_2;
		const float dL_dcos_2 = dL_da * scale_o_blend.z * rscale_sign_blend.x + dL_dr0_2 * (-rscale_o_blend.x * rcos_s_sin_s_2_2);
		const float dL_dsin_2 = dL_da * scale_o_blend.z * rscale_sign_blend.y + dL_dr0_2 * (-rscale_o_blend.y * rcos_s_sin_s_2_2);

		const float rl_1 = 1 / (p_norm);
		const float rl_2 = 1 / (p_norm_2);
		const float rl_3 = 1 / (p_norm_2 * p_norm);
		
		const float dL_dl = dL_du * (2 * a) + dL_dcos_2 * (-2 * p.x * p.x * rl_3) + dL_dsin_2 * (-2 * p.y * p.y * rl_3);

		float3 dL_dx = {0.0f, 0.0f, 0.0f};
		dL_dx.x = p.x * (dL_dl * rl_1 + dL_dcos_2 * 2 * rl_2);
		dL_dx.y = p.y * (dL_dl * rl_1 + dL_dsin_2 * 2 * rl_2);


		if (return_normal){
			// from normal map
			dL_dx.x += dL_point_normal_unnormalized.x * 2 * rscale_sign_blend.x;
			dL_dx.y += dL_point_normal_unnormalized.y * 2 * rscale_sign_blend.y;
#if !DETACH_CURVATURE
			// from curvature map
			dL_dx.x += dL_dcurvature * (- 64 * coeff_1 * coeff_1 * coeff_1 * coeff_2 * p.x * rden_3);
			dL_dx.y += dL_dcurvature * (- 64 * coeff_2 * coeff_2 * coeff_2 * coeff_1 * p.y * rden_3);
#endif
		}

		// splatting to dt
		float dL_dt = dL_dx.x * cam_ray_local.x + dL_dx.y * cam_ray_local.y;
		if (return_depth){
			// distortion to dt
			dL_dt += dL_dmax_t;

			// median depth to dt
			if (contributor == median_contributor) {
				dL_dt += dL_dmedian_depth;
				dL_dweight += dL_dmax_dweight;
			}
			// render depth to dt
			dL_dt += weight * dL_ddepth;
		}

		const double AA = rscale_sign_blend.x * cam_ray_local.x * cam_ray_local.x + 
							   rscale_sign_blend.y * cam_ray_local.y * cam_ray_local.y;
 
		const double BB = 2 * (rscale_sign_blend.x * cam_pos_local.x * cam_ray_local.x + rscale_sign_blend.y * cam_pos_local.y * cam_ray_local.y) - 
							   rscale_sign_blend.z * cam_ray_local.z;
 
		const double CC = rscale_sign_blend.x * cam_pos_local.x * cam_pos_local.x + 
							   rscale_sign_blend.y * cam_pos_local.y * cam_pos_local.y - 
							   rscale_sign_blend.z * cam_pos_local.z;
		const float discriminant = BB * BB - 4 * AA * CC;
		const float r2AA = __frcp_rn(2 * AA);
		float dL_dAA = 0.0f;
		float dL_dBB = 0.0f;
		float dL_dCC = 0.0f;
		float rdiscriminant_sq = 0.0f;
#if QUADRATIC_APPROXIMATION
		if (abs(AA) < 1e-6){
			dL_dAA = 0.0f;
			dL_dBB = dL_dt * (CC * __frcp_rn(BB * BB));
			dL_dCC = dL_dt * (-__frcp_rn(BB));
		}
		else
#endif
		{
			rdiscriminant_sq = __frcp_rn(__fsqrt_rn(discriminant));
			dL_dAA = dL_dt * r2AA * 2 * (- depth - CC * sign * rdiscriminant_sq);
			dL_dBB = dL_dt * (BB * sign * rdiscriminant_sq - 1) * r2AA;
			dL_dCC = dL_dt * (-sign * rdiscriminant_sq);
		}

		float3 dL_dog = {0.0f, 0.0f, 0.0f};
		float3 dL_drg = {0.0f, 0.0f, 0.0f};
		dL_dog.x += dL_dBB * (2 * cam_ray_local.x * rscale_sign_blend.x) + dL_dCC * (2 * cam_pos_local.x * rscale_sign_blend.x);
		dL_dog.y += dL_dBB * (2 * cam_ray_local.y * rscale_sign_blend.y) + dL_dCC * (2 * cam_pos_local.y * rscale_sign_blend.y);
		dL_dog.z += dL_dCC * (-1 * rscale_sign_blend.z);

		dL_drg.x += dL_dAA * (2 * cam_ray_local.x * rscale_sign_blend.x) + dL_dBB * (2 * cam_pos_local.x * rscale_sign_blend.x);
		dL_drg.y += dL_dAA * (2 * cam_ray_local.y * rscale_sign_blend.y) + dL_dBB * (2 * cam_pos_local.y * rscale_sign_blend.y);
		dL_drg.z += dL_dBB * (-1 * rscale_sign_blend.z);

		dL_dog.x += dL_dx.x;
		dL_dog.y += dL_dx.y; 

		dL_drg.x += dL_dx.x * depth;
		dL_drg.y += dL_dx.y * depth;

		float3 dL_dscale_221 = {0.0f, 0.0f, 0.0f};

		// dL_dAA(BB,CC) may introduce more numerical errors.
		const float3 rscale_4_sign = {rscale_sign_blend.x * rscale_o_blend.x, rscale_sign_blend.y * rscale_o_blend.y, rscale_sign_blend.z * rscale_o_blend.z};
		dL_dscale_221.x += dL_da * (-rscale_4_sign.x * cos2_sin2.x * scale_o_blend.z)
					  + dL_dr0_2 * (rscale_o_blend.x * rscale_o_blend.x * cos2_sin2.x * rcos_s_sin_s_2_2);
					//   + dL_dAA * (-rscale_4_sign.x * cam_ray_local.x * cam_ray_local.x)
					//   + dL_dBB * (-2 * rscale_4_sign.x * cam_ray_local.x * cam_pos_local.x)
					//   + dL_dCC * (-rscale_4_sign.x * cam_pos_local.x * cam_pos_local.x);
					  
		dL_dscale_221.y += dL_da * (-rscale_4_sign.y * cos2_sin2.y * scale_o_blend.z)
					  + dL_dr0_2 * (rscale_o_blend.y * rscale_o_blend.y * cos2_sin2.y * rcos_s_sin_s_2_2);
					//   + dL_dAA * (-rscale_4_sign.y * cam_ray_local.y * cam_ray_local.y)
					//   + dL_dBB * (-2 * rscale_4_sign.y * cam_ray_local.y * cam_pos_local.y)
					//   + dL_dCC * (-rscale_4_sign.y * cam_pos_local.y * cam_pos_local.y);
		dL_dscale_221.z += dL_da * (rscale_sign_blend.x * cos2_sin2.x + rscale_sign_blend.y * cos2_sin2.y);
					//   + dL_dBB * (cam_ray_local.z * rscale_4_sign.z)
					//   + dL_dCC * (cam_pos_local.z * rscale_4_sign.z);
					 
		// from normal regularization loss
		if (return_normal){
			dL_dscale_221.x += dL_point_normal_unnormalized.x * (-2 * rscale_4_sign.x * p.x);
			dL_dscale_221.y += dL_point_normal_unnormalized.y * (-2 * rscale_4_sign.y * p.y);
			dL_dscale_221.z += dL_point_normal_unnormalized.z * (rscale_4_sign.z);

			// from curvature map
#if !DETACH_CURVATURE
			dL_dscale_221.x += dL_dcoeff1 * (-rscale_4_sign.x * scale_o_blend.z);
			dL_dscale_221.y += dL_dcoeff2 * (-rscale_4_sign.y * scale_o_blend.z);
			dL_dscale_221.z += dL_dcoeff1 * rscale_sign_blend.x + dL_dcoeff2 * rscale_sign_blend.y;
#endif
		}
		float3 dL_dscale = {dL_dscale_221.x * 2 * scale_o_blend.x, dL_dscale_221.y * 2 * scale_o_blend.y, 0.0f};
		if (!stop_z_gradient)
			dL_dscale.z += dL_dscale_221.z;

		float dL_dview2gaussian_j[16] = {
			 dL_drg.x * ray_point.x, dL_drg.y * ray_point.x, dL_drg.z * ray_point.x, 0,
			 dL_drg.x * ray_point.y, dL_drg.y * ray_point.y, dL_drg.z * ray_point.y, 0,
			 dL_drg.x, dL_drg.y, dL_drg.z, 0,
			 dL_dog.x, dL_dog.y, dL_dog.z, 0
		};
		if (return_normal){
			dL_dview2gaussian_j[0] += point_normal.x * dL_dnormal_x;
			dL_dview2gaussian_j[1] += point_normal.y * dL_dnormal_x;
			dL_dview2gaussian_j[2] += point_normal.z * dL_dnormal_x;
			
			dL_dview2gaussian_j[4] += point_normal.x * dL_dnormal_y;
			dL_dview2gaussian_j[5] += point_normal.y * dL_dnormal_y;
			dL_dview2gaussian_j[6] += point_normal.z * dL_dnormal_y;
			
			dL_dview2gaussian_j[8] += point_normal.x * dL_dnormal_z;
			dL_dview2gaussian_j[9] += point_normal.y * dL_dnormal_z;
			dL_dview2gaussian_j[10] += point_normal.z * dL_dnormal_z;
		}
		// write the gradients to global memory
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 0]), dL_dview2gaussian_j[0]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 1]), dL_dview2gaussian_j[1]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 2]), dL_dview2gaussian_j[2]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 4]), dL_dview2gaussian_j[4]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 5]), dL_dview2gaussian_j[5]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 6]), dL_dview2gaussian_j[6]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 8]), dL_dview2gaussian_j[8]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 9]), dL_dview2gaussian_j[9]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 10]), dL_dview2gaussian_j[10]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 12]), dL_dview2gaussian_j[12]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 13]), dL_dview2gaussian_j[13]);
		atomicAdd(&(dL_dview2gaussian[global_id * 16 + 14]), dL_dview2gaussian_j[14]);
		
		atomicAdd(&(dL_dscales[global_id * 3 + 0]), dL_dscale.x);
		atomicAdd(&(dL_dscales[global_id * 3 + 1]), dL_dscale.y);
		atomicAdd(&(dL_dscales[global_id * 3 + 2]), dL_dscale.z);
		T = test_T;

		// Updating the buffer's sorting.
		for (int i = 1; i < BUFFER_LENGTH; i++){
			sort_ids[i - 1] = sort_ids[i];
			sort_depths[i - 1] = sort_depths[i];
			sort_Gs[i - 1] = sort_Gs[i];
			sort_ps[i - 1] = sort_ps[i];
			sort_r0_2s[i - 1] = sort_r0_2s[i];
			sort_as[i - 1] = sort_as[i];
			sort_ss[i - 1] = sort_ss[i];
			sort_signs[i - 1] = sort_signs[i];
			sort_scales_os[i - 1] = sort_scales_os[i];
			
		}
		sort_depths[BUFFER_LENGTH - 1] = FLT_MAX;
	};

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int all_done = __syncthreads_and(done);
		if (all_done)
			break;
		// Load auxiliary data into shared memory, start in the FRONT!!!
		block.sync();
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			
			for (int ii = 0; ii < 16; ii++)
				 collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
			
			collected_rscales_opacity[block.thread_rank()] = rscales_opacity[coll_id];
			collected_scales[block.thread_rank()] = scales[coll_id];
			collected_scales_sign[block.thread_rank()] = scales_sign[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if (sort_num == BUFFER_LENGTH)
				blend_one();

			if (done)
				break;	

			contributor++;
			float* view2gaussian_j = collected_view2gaussian + j * 16;
			const float3 scale_sign_j = {(float)collected_scales_sign[j].x, (float)collected_scales_sign[j].y, (float)collected_scales_sign[j].z};
			const float3 scale_j = collected_scales[j];
			const float4 rscale_o_j = collected_rscales_opacity[j];
			const float3 rscale_sign_j = {rscale_o_j.x * scale_sign_j.x, rscale_o_j.y * scale_sign_j.y, rscale_o_j.z};
			
			// transform camera center and ray to gaussian's local coordinate system
			// current center is zero
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

			float discriminant = BB * BB - 4 * AA * CC;

			// If the discriminant is less than zero, the ray does not intersect the quadric.
			if (discriminant < 0)
				continue;
			
			float r2AA = __frcp_rn(2 * AA);
			float discriminant_sq_r2AA = sqrt(discriminant) * r2AA;

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
			float sign = -1.0f;
			int AA_sign = copysign(1, AA);
			for(int i = -1; i < 2; i += 2){
#if	QUADRATIC_APPROXIMATION
				if (abs(AA) < 1e-6) // approximation of the intersection equation, see the supplementary material.
					root = __fdividef(-CC, BB);
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
				a = GetParabolaA(cos2_sin2, rscale_sign_j, scale_j);
				s = QuadraticCurveGeodesicDistanceOriginal(p_norm, a);
				s_2 = s * s;
				r0_2 = 1 / (cos2_sin2.x * rscale_o_j.x + cos2_sin2.y * rscale_o_j.y);

				// if the Gaussian weight at the intersection is too small, skip it.
				if (s_2 <= r0_2 * sigma * sigma){
					intersect = true;
					break;
				}
			}

			if (!intersect)
				continue;

			float depth = root;
			if (depth < 0.0f)
				continue;
			
			// float power = -s_2 / (2 * r0_2);
			float power = __fdividef(-s_2, (2.0f * r0_2));
			if (power > 0.0f)
				continue;

			float G = exp(power);
			const float alpha = min(0.99f, rscale_o_j.w * G);

			if (alpha < 1.0f / 255.0f)
				continue;

			int id = collected_id[j];
			float4 scale_o = {scale_j.x, scale_j.y, scale_j.z, rscale_o_j.w};

#pragma unroll
			for(int ii = 0; ii < BUFFER_LENGTH; ii++){
				
				if (depth < sort_depths[ii]){
					swap_T(depth, sort_depths[ii]);
					swap_T(id, sort_ids[ii]);
					swap_T(G, sort_Gs[ii]);
					swap_T(p, sort_ps[ii]);
					swap_T(r0_2, sort_r0_2s[ii]);
					swap_T(a, sort_as[ii]);
					swap_T(s, sort_ss[ii]);
					swap_T(sign, sort_signs[ii]);
					swap_T(scale_o, sort_scales_os[ii]);
				}
			}
			++sort_num;
		}
	}
	if (!done){
		while (sort_num > 0)
			blend_one();
	}
}