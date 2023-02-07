#include <device_launch_parameters.h>

#include <limits>

#include "../Kernel.hpp"
#include "CudaUtils.cuh"
#include "cuda_runtime.h"

cudaStream_t streams[kNumStreams];
bool stream_created = false;

// Global variable
// Need to be registered
// const Point4F* usm_leaf_node_table = nullptr;

__global__ void CudaWarmup() {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

namespace redwood::internal {

void BackendInitialization() {
  CudaWarmup<<<1, 1024>>>();
  HANDLE_ERROR(cudaDeviceSynchronize());
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

void DeviceSynchronize() { HANDLE_ERROR(cudaDeviceSynchronize()); }

void DeviceStreamSynchronize(const int stream_id) {
  HANDLE_ERROR(cudaStreamSynchronize(streams[stream_id]));
}

}  // namespace redwood::internal