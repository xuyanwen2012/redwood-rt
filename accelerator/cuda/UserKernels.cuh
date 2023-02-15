#pragma once

#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cub/cub.cuh>

#include "Redwood/Point.hpp"

namespace cg = cooperative_groups;

__global__ void CudaWarmup() {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

// Modified version
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

inline __device__ float KernelFuncKnn(const Point4F p, const Point4F q) {
  auto dist = float();

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

template <int LeafSize>
__global__ void CudaBarnesHutDebug(const int* u_leaf_indices,
                                   const Point4F* u_lnt_data,
                                   const int* u_lnt_sizes, Point4F q,
                                   float* out, const int num_active_leafs) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  if (tid == 0) {
    float my_total_sum = 0.0f;
    for (int i = 0; i < num_active_leafs; ++i) {
      const auto leaf_id_to_load = u_leaf_indices[i];

      for (int j = 0; j < LeafSize; ++j) {
        const auto p = u_lnt_data[leaf_id_to_load * LeafSize + j];
        my_total_sum += KernelFuncBh(p, q);
      }
    }

    out[0] = my_total_sum;
  }
}

// This 'LeafSize' parameter is important for CUB.
template <int LeafSize>
__global__ void CudaBarnesHut(const int* u_leaf_indices,
                              const Point4F* u_lnt_data, const int* u_lnt_sizes,
                              Point4F q, float* out,
                              const int num_active_leafs) {
  constexpr int warp_threads = 32;
  constexpr int block_threads = 1024;

  constexpr int leaf_size_i_want = LeafSize;

  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  // leaf 256 => 8 per thread
  // ...
  // leaf 64 => 2 per thread
  // leaf 32 => 1 per thread
  constexpr int items_per_thread = leaf_size_i_want / warp_threads;

  constexpr int warps_in_block = block_threads / warp_threads;

  // This is basically 'leaf_size_i_want'
  // constexpr int tile_size = items_per_thread * warp_threads;
  const int warp_id = static_cast<int>(threadIdx.x) / warp_threads;

  using WarpLoad = cub::WarpLoad<Point4F, items_per_thread,
                                 cub::WARP_LOAD_STRIPED, warp_threads>;
  using BlockReduce = cub::BlockReduce<float, block_threads>;

  __shared__ union {
    typename WarpLoad::TempStorage load[warps_in_block];
    BlockReduce::TempStorage reduce;
  } temp_storage;

  float my_total_sum{};

  // 2) Load a single leaf node (per warp, so total of (1024/32=32) warps are
  // loading 32 leafs at a single time).
  // 3) Then Sum all of the loaded value.
  // Note: This block only process (1024/32=32) warps, thus, if the input is
  // 1024 indices (batch_size, num leaf_idx_to_load), then you need to loop
  // this (batch_size) / 32 times.

  // num_active_leafs
  const auto how_many_times_loop = num_active_leafs / warp_threads;

  Point4F thread_data[items_per_thread];
  float thread_value[items_per_thread];

  int it = 0;
  for (; it < how_many_times_loop; ++it) {
    const auto offset = it * warp_threads + warp_id;
    const auto my_leaf_id_to_load = u_leaf_indices[offset];

    // Load entire leaf node at location (given 'leaf_id_to_load')
    WarpLoad(temp_storage.load[warp_id])
        .Load(u_lnt_data + my_leaf_id_to_load * leaf_size_i_want, thread_data);

    for (int i = 0; i < items_per_thread; ++i) {
      thread_value[i] = KernelFuncBh(thread_data[i], q);
    }

    float aggregate = BlockReduce(temp_storage.reduce).Sum(thread_value);

    if (tid == 0) {
      my_total_sum += aggregate;
    }
  }

  // // TODO: fix the reminder issue
  // if (tid == 0) {
  //   for (int i = 0; i < reminder; ++i) {
  //     const auto my_leaf_id_to_load = u_leaf_indices[it * warp_threads +
  //     i]; for (int j = 0; j < leaf_size_i_want; ++j) {
  //       my_total_sum +=
  //           u_lnt_data[my_leaf_id_to_load * leaf_size_i_want + j].data[0];
  //     }
  //   }
  // }

  if (tid == 0) {
    out[0] = my_total_sum;
  }
}

// Debug Kernels are used to check if results are correct.
template <int LeafSize>
__global__ void CudaKnnDebug(const int* u_leaf_indices,  /**/
                             const Point4F* u_q_points,  /**/
                             const int num_active_leafs, /**/
                             float* outs,                /* num buffer * k */
                             const Point4F* u_lnt_data,  /**/
                             const int* u_lnt_sizes) {
  constexpr auto kK = 32;
  // constexpr auto kBufferSize = 1024;

  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  // For debug version, just use each thread to take a leaf (or called
  // /row/item/idx) in the buffer.
  if (tid >= num_active_leafs) return;

  // tid is the index in buffer now
  const auto leaf_id_to_load = u_leaf_indices[tid];
  const auto q = u_q_points[tid];

  // Need to load this from out first.
  float my_mins[kK];
  for (int i = 0; i < kK; ++i) {
    my_mins[i] = outs[tid * kK + i];
  }

  for (int i = 0; i < LeafSize; ++i) {
    const auto p = u_lnt_data[leaf_id_to_load * LeafSize + i];
    const auto dist = KernelFuncKnn(p, q);

    // find min 'idx' first
    int i = 0;
    while (i < kK && dist >= my_mins[i]) {
      i++;
    }

    if (i < kK) {
      // Shift the others back
      for (int j = kK - 1; j > i; j--) {
        my_mins[j] = my_mins[j - 1];
      }
      my_mins[i] = dist;
    }
  }

  // Write back
  for (int i = 0; i < kK; ++i) {
    outs[tid * kK + i] = my_mins[i];
  }
}
