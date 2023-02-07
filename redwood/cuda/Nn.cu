#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

#include "../Kernel.hpp"
#include "CudaUtils.cuh"
#include "cuda_runtime.h"

namespace cg = cooperative_groups;

// I think I can assume there is only 2 streams
extern cudaStream_t streams[kNumStreams];

inline __device__ float KernelFunc(const Point4F p, const Point4F q) {
  auto dist = float();

  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

template <typename DataT, typename QueryT, typename ResultT>
__global__ void NaiveProcessNnBuffer(const QueryT* query_points,
                                     const int* query_idx, const int* leaf_idx,
                                     const DataT* leaf_node_table, ResultT* out,
                                     const int num, const int leaf_node_size) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= num) return;

  // Load all three contents from batch, at index 'tid'
  const auto query_point = query_points[tid];
  const auto q_idx = query_idx[tid];
  const auto leaf_id = leaf_idx[tid];

  auto my_min = 9999999.9f;
  for (int i = 0; i < leaf_node_size; ++i) {
    const auto dist =
        KernelFunc(leaf_node_table[leaf_id * leaf_node_size + i], query_point);

    my_min = min(my_min, dist);
  }

  out[q_idx] = min(out[q_idx], my_min);
}

namespace redwood::internal {

// Main entry to the NN Kernel
void ProcessNnBuffer(const Point4F* query_points,
                     const Point4F* leaf_node_table, const int* query_idx,
                     const int* leaf_idx, float* out, const int num,
                     const int leaf_max_size, const int stream_id) {
  constexpr auto n_blocks = 1u;
  constexpr auto n_threads = 1024u;
  constexpr auto smem_size = 0;
  NaiveProcessNnBuffer<Point4F, Point4F, float>
      <<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          query_points, query_idx, leaf_idx, leaf_node_table, out, num,
          leaf_max_size);
}

}  // namespace redwood::internal