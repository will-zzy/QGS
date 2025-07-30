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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
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
		float* dL_dview2gaussian);

	void preprocess(
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
		const float* view2gaussian,
		const float* view,
		const float* proj,
		const float kernel_size,
		const glm::vec3* campos,
		float3* dL_dmean2D,
		const float* dL_dview2gaussian,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dopacity);
}

#endif