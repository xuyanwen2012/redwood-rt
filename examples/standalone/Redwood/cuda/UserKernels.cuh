#pragma once

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>
#include <limits>

#include "../Point.hpp"

namespace cg = cooperative_groups;

inline __device__ float KernelFuncKnn(const Point4F& p, const Point4F& q) {
  auto dist = float();

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

inline __device__ float KernelFuncBh(const Point4F p, const Point4F q) {
  const auto dx = p.data[0] - q.data[0];
  const auto dy = p.data[1] - q.data[1];
  const auto dz = p.data[2] - q.data[2];
  const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
  const auto inv_dist = rsqrtf(dist_sqr);
  const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
  const auto with_mass = inv_dist3 * p.data[3];
  return dx * with_mass + dy * with_mass + dz * with_mass;
}

// Debug Kernels are used to check if results are correct.
__global__ void CudaNnNaive(const int* u_leaf_indices, /**/
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

template <int LeafSize>
__global__ void CudaNn(const int* u_leaf_indices, /**/
                       const Point4F* u_q_points, /**/
                       const int num_active,      /**/
                       float* u_outs,             /* */
                       const Point4F* u_lnt_data, /**/
                       const int max_leaf_size) {
  constexpr int warp_threads = 32;
  constexpr int block_threads = 1024;
  constexpr int leaf_size_i_want = LeafSize;

  // leaf 256 => 8 per thread
  // ...
  // leaf 64 => 2 per thread
  // leaf 32 => 1 per thread
  constexpr int items_per_thread = leaf_size_i_want / warp_threads;
  constexpr int warps_in_block = block_threads / warp_threads;

  using WarpLoad = cub::WarpLoad<Point4F, items_per_thread,
                                 cub::WARP_LOAD_STRIPED, warp_threads>;

  using WarpReduce = cub::WarpReduce<float>;

  __shared__ union {
    typename WarpLoad::TempStorage load[warps_in_block];
    typename WarpReduce::TempStorage reduce[warps_in_block];
  } temp_storage;

  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  if (tid >= num_active) return;

  const auto how_many_times_loop = num_active / warp_threads;
  // const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;
  const int warp_id = tid / warp_threads;
  const int id_in_warp = tid % warp_threads;

  Point4F thread_data[items_per_thread];
  // float thread_value[items_per_thread];
  float thread_value;

  int it = 0;
  for (; it < how_many_times_loop; ++it) {
    // 'offset' means the index in buffer
    const auto offset = it * warp_threads + warp_id;
    const auto my_leaf_id_to_load = u_leaf_indices[offset];
    const auto q = u_q_points[offset];

    // Load entire leaf node at location (given 'leaf_id_to_load')
    WarpLoad(temp_storage.load[warp_id])
        .Load(u_lnt_data + my_leaf_id_to_load * leaf_size_i_want, thread_data);

    float my_min = 9999999999999.9f;
    for (int i = 0; i < items_per_thread; ++i) {
      thread_value = KernelFuncKnn(thread_data[i], q);
      thread_value = WarpReduce(temp_storage.reduce[warp_id])
                         .Reduce(thread_value, cub::Min());
      my_min = min(my_min, thread_value);
    }

    if (id_in_warp == 0) {
      u_outs[offset] = min(u_outs[offset], my_min);
    }
  }
}

__global__ void FindMinDistWarp6(const Point4F* lnt, const Point4F* u_q,
                                 const int* u_node_idx, float* u_out,
                                 const int num_active,
                                 const int max_leaf_size) {
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
      const auto dist = KernelFuncKnn(
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
