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

#ifndef SIMPLEKNN_H_INCLUDED
#define SIMPLEKNN_H_INCLUDED
#include "spatial.h"

class SimpleKNN
{
public:
	static void knn(int P, float3* points, float* meanDists);
	static void knn_with_indices(int P, float3* points, float* meanDists, int* knn_indices);
};

#endif