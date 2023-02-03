#include <device_launch_parameters.h>

#include <limits>

#include "../Kernel.hpp"
#include "CudaUtils.cuh"
#include "KernelFunc.cuh"
#include "cuda_runtime.h"

// I think I can assume there is only 2 streams
constexpr auto kNumStreams = 2;
cudaStream_t streams[kNumStreams];
bool stream_created = false;

// Global variable
// Need to be registered
const Point4F* usm_leaf_node_table = nullptr;

__global__ void CudaWarmup() {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

namespace redwood::internal {

void BackendInitialization() {}

void DeviceWarmUp() {
  CudaWarmup<<<1, 1024>>>();
  HANDLE_ERROR(cudaDeviceSynchronize());
}

void RegisterLeafNodeTable(const void* leaf_node_table,
                           const int num_leaf_nodes) {
  usm_leaf_node_table = static_cast<const Point4F*>(leaf_node_table);
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
void ProcessNnBuffer(const Point4F* query_points, const int* query_idx,
                     const int* leaf_idx, const Point4F* leaf_node_table,
                     float* out, const int num, const int leaf_max_size,
                     const int stream_id) {
  constexpr auto n_blocks = 1u;
  constexpr auto n_threads = 1024u;
  constexpr auto smem_size = 0;
  NaiveProcessNnBuffer<Point4F, Point4F, float>
      <<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          query_points, query_idx, leaf_idx, usm_leaf_node_table, out, num,
          leaf_max_size, MyFunctor());
}

void DeviceSynchronize() { HANDLE_ERROR(cudaDeviceSynchronize()); }

void DeviceStreamSynchronize(const int stream_id) {
  HANDLE_ERROR(cudaStreamSynchronize(streams[stream_id]));
}

}  // namespace redwood::internal