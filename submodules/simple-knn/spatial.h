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

#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>

#define K_MAX 20

torch::Tensor distCUDA2(const torch::Tensor& points);
torch::Tensor knnWithIndices(const torch::Tensor& points);
std::tuple<torch::Tensor, torch::Tensor> knn_points(const torch::Tensor& points_1, const torch::Tensor& points_2);