#pragma once

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>
#include <limits>

#include "Point.hpp"

namespace cg = cooperative_groups;

__global__ void CudaWarmup() {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

inline __device__ float KernelFuncKnn(const Point4F& p, const Point4F& q) {
  auto dist = float();

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

// Debug Kernels are used to check if results are correct.
__global__ void CudaNnDebug(const int* u_leaf_indices, /**/
                            const Point4F* u_q_points, /**/
                            const int num_active,      /**/
                            float* u_outs,             /* */
                            const Point4F* u_lnt_data, /**/
                            const int max_leaf_size) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  if (tid == 0) {
    for (int i = 0; i < num_active; ++i) {
      const auto leaf_id_to_load = u_leaf_indices[i];
      const auto q = u_q_points[i];
      auto my_min = std::numeric_limits<float>::max();

      for (int j = 0; j < max_leaf_size; ++j) {
        const auto p = u_lnt_data[leaf_id_to_load * max_leaf_size + j];
        const auto dist = KernelFuncKnn(p, q);
        my_min = min(my_min, dist);
      }

      // outs[i] = my_min;
      u_outs[i] = min(u_outs[i], my_min);
    }
  }
}

// Debug Kernels are used to check if results are correct.
__global__ void CudaNaive(const int* u_leaf_indices, /**/
                          const Point4F* u_q_points, /**/
                          const int num_active,      /**/
                          float* u_outs,             /* */
                          const Point4F* u_lnt_data, /**/
                          const int max_leaf_size) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  if (tid < num_active) {
    const auto leaf_id_to_load = u_leaf_indices[tid];
    const auto q = u_q_points[tid];
    auto my_min = std::numeric_limits<float>::max();

    for (int j = 0; j < max_leaf_size; ++j) {
      const auto p = u_lnt_data[leaf_id_to_load * max_leaf_size + j];
      const auto dist = KernelFuncKnn(p, q);
      my_min = min(my_min, dist);
    }

    u_outs[tid] = min(u_outs[tid], my_min);
  }
}

// Parallel *Min* Reduction for Leafs
// template <int LeafSize>
__global__ void FindMinDistWarp6(
    const int* u_leaf_indices,  /* batch_size */
    const Point4F* u_q_points,  /* batch_size */
    const int num_active_leafs, /* how many collected */
    float* outs,                /* batch_size */
    const Point4F* u_lnt_data,  /**/
    const int max_leaf_size) {
  using WarpReduce = cub::WarpReduce<float>;

  __shared__ typename WarpReduce::TempStorage temp_storage[32];
  __shared__ float local_results[1024];
  __shared__ float leaf_node_results[1024];

  auto cta = cg::this_thread_block();
  auto tid = cta.thread_rank();

  constexpr auto warp_size = 32;

  int warp_id = tid / 32;
  int lane_id = tid % 32;
  auto warp = cg::tiled_partition<warp_size>(cta);

  // TODO: numeric max
  leaf_node_results[tid] = 1234.1234f;
  for (int ln = warp_id; ln < num_active_leafs; ln += 32) {
    int ln_id = ln / 32;

    // TODO: numeric max
    local_results[tid] = 9999999.999f;
    const auto leaf_node_uid = u_leaf_indices[ln];
    const auto q = u_q_points[ln];

    for (int group = 0; group < max_leaf_size; group += 32) {
      const int group_id = group / 32;

      // kernel function
      const auto dist = KernelFuncKnn(
          u_lnt_data[leaf_node_uid * max_leaf_size + group + lane_id], q);

      auto my_min = WarpReduce(temp_storage[warp_id]).Reduce(dist, cub::Min());
      my_min = warp.shfl(my_min, 0);
      if (group_id == lane_id) {
        local_results[tid] = my_min;
      }
    }
    auto gl_min = WarpReduce(temp_storage[warp_id])
                      .Reduce(local_results[tid], cub::Min());

    if (lane_id == 0) {
      leaf_node_results[warp_id * 32 + ln_id] = gl_min;
    }
  }

  const auto to_store = leaf_node_results[tid];
  outs[tid] = min(outs[tid], to_store);
}
