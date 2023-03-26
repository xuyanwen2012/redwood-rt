#pragma once

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>
#include <limits>

#include "Functors.cuh"
#include "Redwood/Point.hpp"

namespace cg = cooperative_groups;

// Debug Kernels are used to check if results are correct.
template <typename Functor>
__global__ void CudaNnNaive(const int* u_leaf_indices, /**/
                            const Point4F* u_q_points, /**/
                            const int num_active,      /**/
                            float* u_outs,             /* */
                            const Point4F* u_lnt_data, /**/
                            const int max_leaf_size, const Functor functor) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  if (tid < num_active) {
    const auto leaf_id_to_load = u_leaf_indices[tid];
    const auto q = u_q_points[tid];
    auto my_min = std::numeric_limits<float>::max();

    for (int j = 0; j < max_leaf_size; ++j) {
      const auto p = u_lnt_data[leaf_id_to_load * max_leaf_size + j];
      const auto dist = functor(p, q);
      my_min = min(my_min, dist);
    }

    u_outs[tid] = min(u_outs[tid], my_min);
  }
}

template <typename Functor>
__global__ void FindMinDistWarp6(const Point4F* lnt, const Point4F* u_q,
                                 const int* u_node_idx, float* u_out,
                                 const int num_active, const int max_leaf_size,
                                 const Functor functor) {
  using WarpReduce = cub::WarpReduce<float>;

  __shared__ WarpReduce::TempStorage temp_storage[32];
  __shared__ float local_results[1024];
  __shared__ float leaf_node_results[1024];

  auto cta = cg::this_thread_block();
  auto tid = cta.thread_rank();
  constexpr auto warp_size = 32u;

  int warp_id = tid / 32;
  int lane_id = tid % 32;
  auto warp = cg::tiled_partition<warp_size>(cta);

  leaf_node_results[tid] = std::numeric_limits<float>::max();
  int cached_query_idx;
  for (int ln = warp_id; ln < num_active; ln += 32) {
    int ln_id = ln / 32;

    local_results[tid] = std::numeric_limits<float>::max();
    const auto leaf_node_uid = u_node_idx[ln];
    const auto query_data = u_q[ln];

    if (ln_id == lane_id) {
      cached_query_idx = ln;
    }

    for (int group = 0; group < max_leaf_size; group += 32) {
      const int group_id = group / 32;

      // kernel function
      const auto dist = functor(
          lnt[leaf_node_uid * max_leaf_size + group + lane_id], query_data);

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

  auto to_store = leaf_node_results[tid];
  u_out[cached_query_idx] = min(u_out[cached_query_idx], to_store);
}
