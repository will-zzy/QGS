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

 #include "spatial.h"
 #include "simple_knn.h"
 
 torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);
 
  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);

  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());
 
  return means;
}
 
torch::Tensor
knnWithIndices(const torch::Tensor& points)
{
  const int P = points.size(0);
  auto float_opts = points.options().dtype(torch::kFloat32);
  auto int_opts = points.options().dtype(torch::kInt32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  torch::Tensor knn_indices = torch::full({P, K_MAX}, 0.0, int_opts);
  SimpleKNN::knn_with_indices(
    P, 
    (float3*)points.contiguous().data<float>(), 
    means.contiguous().data<float>(), 
    knn_indices.contiguous().data<int>()
  );
  return knn_indices;
  // return means;
}

std::tuple<torch::Tensor, torch::Tensor>
knn_points(const torch::Tensor& points_1, const torch::Tensor& points_2)
{
  const int P1 = points_1.size(0);
  const int P2 = points_2.size(0);
  torch::Tensor points_all = torch::cat({points_2, points_1}, 0);
  auto float_opts = points_all.options().dtype(torch::kFloat32);
  auto int_opts = points_all.options().dtype(torch::kInt32);
  torch::Tensor knn_indices = torch::full({P1, K_MAX}, 0.0, int_opts); // for each point in P1, K-nearest points in P2
  torch::Tensor dists = torch::full({P1, K_MAX}, 0.0, float_opts); // knn distances
  SimpleKNN::knn_points(
    P1, P2, 
    (float3*)points_all.contiguous().data<float>(), 
    knn_indices.contiguous().data<int>(),
    dists.contiguous().data<float>()
  );
  return std::make_tuple(knn_indices, dists);
}
