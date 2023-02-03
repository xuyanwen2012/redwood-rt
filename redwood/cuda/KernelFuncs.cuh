
#pragma once

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"

namespace cg = cooperative_groups;

struct MyFunctor {
  inline __device__ float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += diff * diff;
    }

    return sqrtf(dist);
  }
};

template <typename DataT, typename QueryT, typename ResultT>
__global__ void NaiveProcessNnBuffer(const QueryT* query_points,
                                     const int* query_idx, const int* leaf_idx,
                                     const DataT* leaf_node_table, ResultT* out,
                                     const int num, const int leaf_node_size,
                                     MyFunctor functor) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num) return;

  // Load all three contents from batch, at index 'tid'
  const auto query_point = query_points[tid];
  const auto q_idx = query_idx[tid];
  const auto leaf_id = leaf_idx[tid];

  auto my_min = 9999999.9f;
  for (int i = 0; i < leaf_node_size; ++i) {
    const auto dist =
        functor(leaf_node_table[leaf_id * leaf_node_size + i], query_point);

    my_min = min(my_min, dist);
  }

  out[q_idx] = min(out[q_idx], my_min);

  // printf("[%d] %f\n", tid, my_min);
}
