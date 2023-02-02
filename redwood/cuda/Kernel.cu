#include <device_launch_parameters.h>

#include <limits>

#include "../Kernel.hpp"
#include "CudaUtils.cuh"
#include "cuda_runtime.h"

// I think I can assume there is only 2 streams
constexpr auto kNumStreams = 2;
cudaStream_t streams[kNumStreams];
bool stream_created = false;

// Global variable
// Need to be registered
const float* usm_leaf_node_table = nullptr;

__global__ void CudaWarmup() {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

struct MyFunctor {
  // GPU version
  __device__ float operator()(const float p, const float q) const {
    auto dist = float();

    const auto diff = p - q;
    dist += diff * diff;

    return sqrtf(dist);
  }
};

__global__ void NaiveProcessNnBuffer(const float* query_points,
                                     const int* query_idx, const int* leaf_idx,
                                     const float* leaf_node_table, float* out,
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

namespace redwood::internal {
void DeviceWarmUp() {
  CudaWarmup<<<1, 1024>>>();
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void RegisterLeafNodeTable(const void* leaf_node_table,
                           const int num_leaf_nodes) {
  usm_leaf_node_table = static_cast<const float*>(leaf_node_table);
}

// CUDA Only
void AttachStreamMem(const int stream_id, void* addr) {
  if (!stream_created) {
    for (unsigned i = 0; i < kNumStreams; i++) {
      HANDLE_ERROR(cudaStreamCreate(&streams[i]));
    }
    stream_created = true;
  }

  cudaStreamAttachMemAsync(streams[stream_id], addr);
}

// Main entry to the NN Kernel
void ProcessNnBuffer(const float* query_points, const int* query_idx,
                     const int* leaf_idx, const float* leaf_node_table,
                     float* out, const int num, const int stream_id) {
  constexpr auto n_blocks = 1u;
  constexpr auto n_threads = 1024u;
  constexpr auto smem_size = 0;
  NaiveProcessNnBuffer<<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
      query_points, query_idx, leaf_idx, usm_leaf_node_table, out, num, 32,
      MyFunctor());
}

void DeviceSynchronize() { HANDLE_ERROR(cudaDeviceSynchronize()); }

void DeviceStreamSynchronize(const int stream_id) {
  HANDLE_ERROR(cudaStreamSynchronize(streams[stream_id]));
}

}  // namespace redwood::internal