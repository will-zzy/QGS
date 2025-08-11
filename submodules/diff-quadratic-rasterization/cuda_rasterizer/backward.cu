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

#include "backward.h"
#include "auxiliary.h"
#include "stopthepop_QGS/resorted_render.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);
 
	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
 
	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;
 
	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;
 
	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;
 
	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
	 	float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;
 
		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];
 
		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
 
			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;
 
			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];
 
			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;
 
				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));
 
				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);
 
				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}
 
	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));
 
	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });
 
	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}
 
// Backward method for creating a view to gaussian coordinate system transformation matrix
__device__ void computeView2Gaussian_backward(
	const float focal_x, const float focal_y,
	int idx, 
	const float3& mean, 
	const glm::vec4 rot, 
	const float* viewmatrix,  
	const float* view2gaussian, 
	const float* dL_dview2gaussian,
	const float W, const float H,
	const bool reciprocal_z,
	glm::vec3* dL_dmeans, 
	glm::vec4* dL_drots,
	float3& dL_dmean2D
)
{
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

 
	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f
	);
 
	glm::mat4 W2V = glm::mat4(
		 viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		 viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		 viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		 viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);
 
	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;
 
 
	// compute the gradient here
	// G2V = [R, t], V2G = inverse(G2V) = [R^T, -R^T * t]
	// V2G_R = G2V_R^T
	// V2G_t = -G2V_R^T * G2V_t
	glm::mat3 G2V_R_t = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);
	glm::mat3 G2V_R = glm::transpose(G2V_R_t);
	glm::vec3 G2V_t = glm::vec3(
		G2V[3][0], G2V[3][1], G2V[3][2]
	);
 
	// dL_dG2V_R = dL_dV2G_R^T
	// dL_dG2V_t = -dL_dV2G_t * G2V_R^T
	glm::mat3 dL_dV2G_R_t = glm::mat3(
		 dL_dview2gaussian[0], dL_dview2gaussian[4], dL_dview2gaussian[8],
		 dL_dview2gaussian[1], dL_dview2gaussian[5], dL_dview2gaussian[9],
		 dL_dview2gaussian[2], dL_dview2gaussian[6], dL_dview2gaussian[10]
	);
	glm::vec3 dL_dV2G_t = glm::vec3(
		 dL_dview2gaussian[12], dL_dview2gaussian[13], dL_dview2gaussian[14]
	);
 
	// also gradient from -R^T * t
	glm::mat3 dL_dG2V_R_from_t = glm::mat3(
		 -dL_dV2G_t.x * G2V_t.x, -dL_dV2G_t.x * G2V_t.y, -dL_dV2G_t.x * G2V_t.z,
		 -dL_dV2G_t.y * G2V_t.x, -dL_dV2G_t.y * G2V_t.y, -dL_dV2G_t.y * G2V_t.z,
		 -dL_dV2G_t.z * G2V_t.x, -dL_dV2G_t.z * G2V_t.y, -dL_dV2G_t.z * G2V_t.z
	);

	glm::mat3 dL_dG2V_R = dL_dV2G_R_t + dL_dG2V_R_from_t;
	glm::vec3 dL_dG2V_t = -dL_dV2G_t * G2V_R_t;

	// The centroid depth of the primitive.
	float p_view_z = G2V[3][2]; 
	float factor = max(p_view_z, 0.1f);

	if (reciprocal_z)
		factor = factor + 1 / factor;

	// This approach draws upon the gradient projection strategy employed by 2DGS.
	dL_dmean2D.x = dL_dG2V_t.x * factor * W; 
	dL_dmean2D.y = dL_dG2V_t.y * factor * H;
	
	// dL_dG2V = [dL_dG2V_R, dL_dG2V_t]
	glm::mat4 dL_dG2V = glm::mat4(
		dL_dG2V_R[0][0], dL_dG2V_R[0][1], dL_dG2V_R[0][2], 0.0f,
		dL_dG2V_R[1][0], dL_dG2V_R[1][1], dL_dG2V_R[1][2], 0.0f,
		dL_dG2V_R[2][0], dL_dG2V_R[2][1], dL_dG2V_R[2][2], 0.0f,
		dL_dG2V_t.x, dL_dG2V_t.y, dL_dG2V_t.z, 0.0f
	);

	// Gaussian to view transform
	glm::mat4 dL_dG2W = glm::transpose(W2V) * dL_dG2V;
 
	
	// Gaussian to world transform
	// dL_dG2W_R = dL_dG2W_R^T
	// dL_dG2W_t = dL_dG2W_t
	glm::mat3 dL_dG2W_R = glm::mat3(
		dL_dG2W[0][0], dL_dG2W[0][1], dL_dG2W[0][2],
		dL_dG2W[1][0], dL_dG2W[1][1], dL_dG2W[1][2],
		dL_dG2W[2][0], dL_dG2W[2][1], dL_dG2W[2][2]
	);
	glm::vec3 dL_dG2W_t = glm::vec3(
		 dL_dG2W[3][0], dL_dG2W[3][1], dL_dG2W[3][2]
	);
	glm::mat3 dL_dR = dL_dG2W_R;
 
	// Gradients of loss w.r.t. means
	glm::vec3* dL_dmean = dL_dmeans + idx;
	dL_dmean->x = dL_dG2W_t.x;
	dL_dmean->y = dL_dG2W_t.y;
	dL_dmean->z = dL_dG2W_t.z;

	glm::mat3 dL_dMt = dL_dR;
 
	// // Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);
 
	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}


// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float W, const float H,
	const bool reciprocal_z,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float focal_x, const float focal_y,
	const float* proj,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* view2gaussian,
	const float* viewmatrix,
	const float* dL_dview2gaussian,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		 return;
	
	// compute the gradient of view2gaussian
	computeView2Gaussian_backward(focal_x, focal_y, idx, means[idx], rotations[idx], viewmatrix, view2gaussian + 16 * idx, dL_dview2gaussian + 16 * idx, W, H, reciprocal_z, dL_dmeans, dL_drot, dL_dmean2D[idx]);
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		 computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
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

	const bool inside = pix.x < W&& pix.y < H;
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
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	
	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_C = inside ? final_Ts[pix_id + 4 * H * W] : 0;
	const float final_C2 = inside ? final_Ts[pix_id + 5 * H * W] : 0;
	const float final_A = 1 - T_final;
	// gradient from normalization
 
	float last_dL_dT = 0;
	
	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
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
	if (inside){
		for (int i = 0; i < C; i++){
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
		}
			
		for (int i = 0; i < 3; i++)
			dL_dnormal2D[i] = dL_dpixels[(C+i) * H * W + pix_id];

		dL_ddepth = dL_dpixels[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_dpixels[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_dpixels[DISTORTION_OFFSET * H * W + pix_id];
		dL_dmedian_depth = dL_dpixels[MIDDEPTH_OFFSET * H * W + pix_id];
		dL_dmax_dweight = dL_dpixels[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
		dL_dcurv = dL_dpixels[CURVATURE_OFFSET * H * W + pix_id];
		dL_dreg_curv = dL_dpixels[CURV_DISTORTION_OFFSET * H * W + pix_id];
	}
	
	float last_alpha = 0;
	float last_depth = 0;
	float last_curv = 0;
	float last_color[C] = { 0 };
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_curv_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	float accum_curvature_rec = 0;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			for (int i = 0; i < C; i++)
				 collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			
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
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;
			
			float* view2gaussian_j = collected_view2gaussian + j * 16;

 			const float3 scale_sign_j = {(float)collected_scales_sign[j].x, (float)collected_scales_sign[j].y, (float)collected_scales_sign[j].z};
 			const float3 scale_j = collected_scales[j];
 			const float4 rscale_o_j = collected_rscales_opacity[j];
 			const float3 rscale_sign_j = {rscale_o_j.x * scale_sign_j.x, rscale_o_j.y * scale_sign_j.y, rscale_o_j.z};
 			
			// transform camera center and ray to gaussian's local coordinate system
			float3 cam_pos_local = {view2gaussian_j[12], view2gaussian_j[13], view2gaussian_j[14]};
			float3 cam_ray_local = transformPoint4x3_without_t(ray_point, view2gaussian_j);

			// compute the minimal value
			// use AA, BB, CC so that the name is unique
			const double AA = rscale_sign_j.x * cam_ray_local.x * cam_ray_local.x + 
							   rscale_sign_j.y * cam_ray_local.y * cam_ray_local.y;
 
			const double BB = 2 * (rscale_sign_j.x * cam_pos_local.x * cam_ray_local.x + rscale_sign_j.y * cam_pos_local.y * cam_ray_local.y) - 
							   rscale_sign_j.z * cam_ray_local.z;
 
			const double CC = rscale_sign_j.x * cam_pos_local.x * cam_pos_local.x + 
							   rscale_sign_j.y * cam_pos_local.y * cam_pos_local.y - 
							   rscale_sign_j.z * cam_pos_local.z;

			float discriminant = BB * BB - 4 * AA * CC;

			if (discriminant < 0)
				continue;

			float r2AA = __frcp_rn(2 * AA);
			float discriminant_sq_r2AA = sqrt(discriminant) * r2AA;

 
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
			float px_2 = 0.0f;
			float py_2 = 0.0f;
			int AA_sign = copysign(1, AA);
			#pragma unroll
			for(int i = -1; i < 2; i += 2){
#if	QUADRATIC_APPROXIMATION
				if (abs(AA) < 1e-6)
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
				px_2 = p.x * p.x;
				py_2 = p.y * p.y;
				
				p_norm_2 = px_2 + py_2 + 1e-7;
				p_norm = __fsqrt_rn(p_norm_2);
				cos2_sin2 = {__fdividef(px_2, p_norm_2), __fdividef(py_2, p_norm_2)};
				a = GetParabolaA(cos2_sin2, rscale_sign_j, scale_j);
				s = QuadraticCurveGeodesicDistanceOriginal(p_norm, a);
				s_2 = s * s;
				r0_2 = __fdividef(1.0f, (cos2_sin2.x * rscale_o_j.x + cos2_sin2.y * rscale_o_j.y));

				if (s_2 <= r0_2 * sigma * sigma){
					intersect = true;
					break;
				}
			}

			if (!intersect)
				continue;
 
			float power = __fdividef(-s_2, (2.0f * r0_2));
			float depth = root;

			const float G = __expf(power);
			const float alpha = min(0.99f, rscale_o_j.w * G);

			if (alpha < 1.0f / 255.0f)
				continue;

			T = __fdividef(T, 1.f - alpha);
			const float weight = alpha * T;
 
			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			const float one_minus_last_alpha = 1.f - last_alpha;
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + one_minus_last_alpha * accum_rec[ch];
				last_color[ch] = c;
 
				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), weight * dL_dchannel);
			}
			
			
			float dL_dweight = 0.0f;
#if !DETACH_CURVATURE
			float coeff_1 = scale_j.z * rscale_sign_j.x;
			float coeff_2 = scale_j.z * rscale_sign_j.y;
			float a2u2 = coeff_1 * coeff_1 * px_2;
			float b2v2 = coeff_2 * coeff_2 * py_2;
			float den = 1 + 4 * (a2u2 + b2v2);

			// gaussian curvature
			// float curvature = 4 * coeff_1 * coeff_2 * __frcp_rn(den * den);
			float curvature = __fdividef(4 * coeff_1 * coeff_2, den * den);
			
			// curvature distortion to weight and curvature
			dL_dweight += (final_C2 + curvature * curvature * final_A - 2 * curvature * final_C) * dL_dreg_curv;
			float dL_dcurvature = 2.0f * (T * alpha) * (curvature * final_A - final_C) * dL_dreg_curv;

			const float rden_3 = __frcp_rn(den * den * den);
			const float dL_dcoeff1 = (__fdividef(curvature, coeff_1) - (64 * a2u2 * coeff_2) * rden_3) * dL_dcurvature;
			const float dL_dcoeff2 = (__fdividef(curvature, coeff_2) - (64 * b2v2 * coeff_1) * rden_3) * dL_dcurvature;
			
			// curvature map to alpha 
			accum_curv_rec = last_alpha * last_curv + one_minus_last_alpha * accum_curv_rec;
			last_curv = curvature;
			dL_dalpha += (curvature - accum_curv_rec) * dL_dcurv;
			
			// curvature map to curvature
			dL_dcurvature += weight * dL_dcurv;
#endif
			// depth distortion
			// distortion to weight, alpha
			const float max_t = depth;
			const float mapped_max_t = (FAR_PLANE * max_t - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * max_t);
			float dmax_t_dd = (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * max_t * max_t);
			dL_dweight += (final_D2 + mapped_max_t * mapped_max_t * final_A - 2 * mapped_max_t * final_D) * dL_dreg;			
#if DETACH_DIST2WEIGHT
			dL_dalpha += 0.0f;
#else
			dL_dalpha += dL_dweight - last_dL_dT;
#endif
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT; // tmp

			// distortion to depth
			float dL_dmax_t = 2.0f * (T * alpha) * (mapped_max_t * final_A - final_D) * dL_dreg * dmax_t_dd;
			
			// depth map to alpha
			accum_depth_rec = last_alpha * last_depth + one_minus_last_alpha * accum_depth_rec;
			last_depth = max_t;
			dL_dalpha = __fmaf_rn((max_t - accum_depth_rec), dL_ddepth, dL_dalpha);

			// alpha map to alpha
			accum_alpha_rec = __fmaf_rn(one_minus_last_alpha, accum_alpha_rec, last_alpha * 1.f);
			dL_dalpha = __fmaf_rn((1.f - accum_alpha_rec), dL_daccum, dL_dalpha);

			// normal regularzation
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

			
			float sum_sq = __fmaf_rn(point_normal_unnormalized.x, point_normal_unnormalized.x,
						   __fmaf_rn(point_normal_unnormalized.y, point_normal_unnormalized.y,
						   __fmaf_rn(point_normal_unnormalized.z, point_normal_unnormalized.z, 1e-7f)));
 			float length = __fsqrt_rn(sum_sq);
			const float rlength = __frcp_rn(length);
			float3 point_normal = { point_normal_unnormalized.x * rlength, 
									point_normal_unnormalized.y * rlength, 
									point_normal_unnormalized.z * rlength};

			float nx = __fmaf_rn(view2gaussian_j[0], point_normal.x,
					   __fmaf_rn(view2gaussian_j[1], point_normal.y,
							  	 view2gaussian_j[2] * point_normal.z));
			float ny = __fmaf_rn(view2gaussian_j[4], point_normal.x,
					   __fmaf_rn(view2gaussian_j[5], point_normal.y,
							  	 view2gaussian_j[6] * point_normal.z));
			float nz = __fmaf_rn(view2gaussian_j[8], point_normal.x,
					   __fmaf_rn(view2gaussian_j[9], point_normal.y,
							  	 view2gaussian_j[10]* point_normal.z));

			float3 normal = {nx, ny, nz};
			float dL_dnormal_reg[3] = {0};
			float normal_tmp[3] = {normal.x, normal.y, normal.z};
			// // Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				// normal map to alpha
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + one_minus_last_alpha * accum_normal_rec[ch];
				last_normal[ch] = normal_tmp[ch];
				dL_dalpha += (normal_tmp[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				
				// normal map to normal
				dL_dnormal_reg[ch] = weight * dL_dnormal2D[ch]; 
			}
			float dL_dnormal_x = dL_dnormal_reg[0];
			float dL_dnormal_y = dL_dnormal_reg[1];
			float dL_dnormal_z = dL_dnormal_reg[2];
 
			const float3 dL_dpoint_normal = { 
				view2gaussian_j[0] * dL_dnormal_x + view2gaussian_j[4] * dL_dnormal_y + view2gaussian_j[8] * dL_dnormal_z,
				view2gaussian_j[1] * dL_dnormal_x + view2gaussian_j[5] * dL_dnormal_y + view2gaussian_j[9] * dL_dnormal_z,
				view2gaussian_j[2] * dL_dnormal_x + view2gaussian_j[6] * dL_dnormal_y + view2gaussian_j[10] * dL_dnormal_z
			};

			float dL_dlength = dL_dpoint_normal.x * point_normal_unnormalized.x + dL_dpoint_normal.y * point_normal_unnormalized.y + dL_dpoint_normal.z * point_normal_unnormalized.z;
			dL_dlength *= -1.f / (length * length);
			const float3 dL_point_normal_unnormalized = {
				(dL_dpoint_normal.x + dL_dlength * point_normal_unnormalized.x) * rlength * (-sign_normal),
				(dL_dpoint_normal.y + dL_dlength * point_normal_unnormalized.y) * rlength * (-sign_normal),
				(dL_dpoint_normal.z + dL_dlength * point_normal_unnormalized.z) * rlength * (-sign_normal)
			};
			dL_dalpha *= T;

			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;
 
			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += __fdividef(-T_final, 1.f - alpha) * bg_dot_dpixel;

			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

			const float dL_dG = rscale_o_j.w * dL_dalpha;
			const float dL_ds = dL_dG * __fdividef(-G * s, r0_2);
			const float dL_dr0_2 = dL_dG * (G * __fdividef(-power, r0_2));
			const float u = 2 * a * p_norm;
			const float dL_du = dL_ds * __fdividef(__fsqrt_rn(u * u + 1), (2 * a));
			float dL_da = 2 * dL_du * p_norm - dL_ds * __fdividef(s, a);
			float rcos_s_sin_s_2_2 = r0_2 * r0_2;
			const float dL_dcos_2 = dL_da * scale_j.z * rscale_sign_j.x + dL_dr0_2 * (-rscale_o_j.x * rcos_s_sin_s_2_2);
			const float dL_dsin_2 = dL_da * scale_j.z * rscale_sign_j.y + dL_dr0_2 * (-rscale_o_j.y * rcos_s_sin_s_2_2);

			const float rl_1 = __frcp_rn(p_norm);
			const float rl_2 = __frcp_rn(p_norm_2);
			const float rl_3 = __frcp_rn(p_norm_2 * p_norm);
			
			const float dL_dl = dL_du * (2 * a) + dL_dcos_2 * (-2 * px_2 * rl_3) + dL_dsin_2 * (-2 * py_2 * rl_3);
			float3 dL_dx = {0.0f, 0.0f, 0.0f};
			dL_dx.x = p.x * (dL_dl * rl_1 + dL_dcos_2 * 2 * rl_2);
			dL_dx.y = p.y * (dL_dl * rl_1 + dL_dsin_2 * 2 * rl_2);
			// from normal map
			dL_dx.x += dL_point_normal_unnormalized.x * 2 * rscale_sign_j.x;
			dL_dx.y += dL_point_normal_unnormalized.y * 2 * rscale_sign_j.y;

			// from curvature map
#if !DETACH_CURVATURE
			dL_dx.x += dL_dcurvature * (- 64 * coeff_1 * coeff_1 * coeff_1 * coeff_2 * p.x * rden_3);
			dL_dx.y += dL_dcurvature * (- 64 * coeff_2 * coeff_2 * coeff_2 * coeff_1 * p.y * rden_3);
#endif
			// splatting to dt
			float dL_dt = dL_dx.x * cam_ray_local.x + dL_dx.y * cam_ray_local.y;
			// distortion to dt
			dL_dt += dL_dmax_t;

			// median depth to dt
			if (contributor == median_contributor - 1) {
				dL_dt += dL_dmedian_depth;
				dL_dweight += dL_dmax_dweight;
			}
			// render depth to dt
			dL_dt += weight * dL_ddepth;


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
				dL_dAA = dL_dt * r2AA * 2 * (- root - CC * sign * rdiscriminant_sq);
				dL_dBB = dL_dt * (BB * sign * rdiscriminant_sq - 1) * r2AA;
				dL_dCC = dL_dt * (-sign * rdiscriminant_sq);
			}

			float3 dL_dog = {0.0f, 0.0f, 0.0f};
			float3 dL_drg = {0.0f, 0.0f, 0.0f};
			dL_dog.x += dL_dBB * (2 * cam_ray_local.x * rscale_sign_j.x) + dL_dCC * (2 * cam_pos_local.x * rscale_sign_j.x);
			dL_dog.y += dL_dBB * (2 * cam_ray_local.y * rscale_sign_j.y) + dL_dCC * (2 * cam_pos_local.y * rscale_sign_j.y);
			dL_dog.z += dL_dCC * (-1 * rscale_sign_j.z);
 
			dL_drg.x += dL_dAA * (2 * cam_ray_local.x * rscale_sign_j.x) + dL_dBB * (2 * cam_pos_local.x * rscale_sign_j.x);
			dL_drg.y += dL_dAA * (2 * cam_ray_local.y * rscale_sign_j.y) + dL_dBB * (2 * cam_pos_local.y * rscale_sign_j.y);
			dL_drg.z += dL_dBB * (-1 * rscale_sign_j.z);

			dL_dog.x += dL_dx.x;
			dL_dog.y += dL_dx.y; 

			dL_drg.x += dL_dx.x * root;
			dL_drg.y += dL_dx.y * root;

			float3 dL_dscale_221 = {0.0f, 0.0f, 0.0f};

			// dL_dAA(BB,CC) may introduce more numerical errors.
			const float3 rscale_4_sign = {rscale_sign_j.x * rscale_o_j.x, rscale_sign_j.y * rscale_o_j.y, rscale_sign_j.z * rscale_o_j.z};
			dL_dscale_221.x += dL_da * (-rscale_4_sign.x * cos2_sin2.x * scale_j.z)
						  + dL_dr0_2 * (rscale_o_j.x * rscale_o_j.x * cos2_sin2.x * rcos_s_sin_s_2_2);
						  + dL_dAA * (-rscale_4_sign.x * cam_ray_local.x * cam_ray_local.x)
						  + dL_dBB * (-2 * rscale_4_sign.x * cam_ray_local.x * cam_pos_local.x)
						  + dL_dCC * (-rscale_4_sign.x * cam_pos_local.x * cam_pos_local.x);
						  
			dL_dscale_221.y += dL_da * (-rscale_4_sign.y * cos2_sin2.y * scale_j.z)
						  + dL_dr0_2 * (rscale_o_j.y * rscale_o_j.y * cos2_sin2.y * rcos_s_sin_s_2_2);
						  + dL_dAA * (-rscale_4_sign.y * cam_ray_local.y * cam_ray_local.y)
						  + dL_dBB * (-2 * rscale_4_sign.y * cam_ray_local.y * cam_pos_local.y)
						  + dL_dCC * (-rscale_4_sign.y * cam_pos_local.y * cam_pos_local.y);
 
			dL_dscale_221.z += dL_da * (rscale_sign_j.x * cos2_sin2.x + rscale_sign_j.y * cos2_sin2.y);
						  + dL_dBB * (cam_ray_local.z * rscale_4_sign.z)
						  + dL_dCC * (cam_pos_local.z * rscale_4_sign.z);	 
			// from normal regularization loss
			dL_dscale_221.x += dL_point_normal_unnormalized.x * (-2 * rscale_4_sign.x * p.x);
			dL_dscale_221.y += dL_point_normal_unnormalized.y * (-2 * rscale_4_sign.y * p.y);
			dL_dscale_221.z += dL_point_normal_unnormalized.z * (rscale_4_sign.z);

#if !DETACH_CURVATURE
			// from curvature map
			dL_dscale_221.x += dL_dcoeff1 * (-rscale_4_sign.x * scale_j.z);
			dL_dscale_221.y += dL_dcoeff2 * (-rscale_4_sign.y * scale_j.z);
			dL_dscale_221.z += dL_dcoeff1 * rscale_sign_j.x + dL_dcoeff2 * rscale_sign_j.y;
#endif
			float3 dL_dscale = {dL_dscale_221.x * 2 * scale_j.x, dL_dscale_221.y * 2 * scale_j.y, 0.0f};
			if (!stop_z_gradient)
				dL_dscale.z += dL_dscale_221.z;		 

			float dL_dview2gaussian_j[16] = {
				 dL_drg.x * ray_point.x, dL_drg.y * ray_point.x, dL_drg.z * ray_point.x, 0,
				 dL_drg.x * ray_point.y, dL_drg.y * ray_point.y, dL_drg.z * ray_point.y, 0,
				 dL_drg.x, dL_drg.y, dL_drg.z, 0,
				 dL_dog.x, dL_dog.y, dL_dog.z, 0
			};

			dL_dview2gaussian_j[0] += point_normal.x * dL_dnormal_x;
			dL_dview2gaussian_j[1] += point_normal.y * dL_dnormal_x;
			dL_dview2gaussian_j[2] += point_normal.z * dL_dnormal_x;
			
			dL_dview2gaussian_j[4] += point_normal.x * dL_dnormal_y;
			dL_dview2gaussian_j[5] += point_normal.y * dL_dnormal_y;
			dL_dview2gaussian_j[6] += point_normal.z * dL_dnormal_y;
			
			dL_dview2gaussian_j[8] += point_normal.x * dL_dnormal_z;
			dL_dview2gaussian_j[9] += point_normal.y * dL_dnormal_z;
			dL_dview2gaussian_j[10] += point_normal.z * dL_dnormal_z;

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
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float W, const float H,
	const bool reciprocal_z,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float focal_x, const float focal_y,
	const float* view2gaussian,
	const float* viewmatrix,
	const float* projmatrix,
	const float kernel_size,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* dL_dview2gaussian,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dopacity)
{

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		W, H,
		reciprocal_z,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		focal_x, focal_y,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		view2gaussian,
		viewmatrix,
		dL_dview2gaussian,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float principal_x, const float principal_y,
	const float sigma,
	const bool stop_z_gradient,
	const float* bg_color,
	const float4* rscales_opacity,
	const int3* scales_sign,
	const float* colors,
	const float* view2gaussian,
	const float* viewmatrix,
	const float3* means3D,
	const float3* scales,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* out_colors,
	const float* dL_dpixels,
	const bool return_depth,
	const bool return_normal,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dscales,
	float* dL_dview2gaussian)
{
#if PIXEL_RESORTING
	renderkBufferBackwardCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		principal_x, principal_y,
		sigma,
		stop_z_gradient,
		bg_color,
		rscales_opacity,
		scales_sign,
		colors,
		view2gaussian,
		viewmatrix,
		means3D,
		scales,
		depths,
		final_Ts,
		n_contrib,
		out_colors,
		dL_dpixels,
		return_depth,
		return_normal,
		dL_dopacity,
		dL_dcolors,
		dL_dscales,
		dL_dview2gaussian
	);
#else
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		principal_x, principal_y,
		sigma,
		stop_z_gradient,
		bg_color,
		rscales_opacity,
		scales_sign,
		colors,
		view2gaussian,
		viewmatrix,
		means3D,
		scales,
		depths,
		final_Ts,
		n_contrib,
		dL_dpixels,
		return_depth,
		return_normal,
		dL_dopacity,
		dL_dcolors,
		dL_dscales,
		dL_dview2gaussian
		);
#endif
}