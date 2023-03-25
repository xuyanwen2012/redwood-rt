#include <iostream>

#include "Redwood.hpp"

//
#include "CudaUtils.cuh"
#include "Kernels.hpp"
#include "UserKernels.cuh"

namespace redwood {

constexpr auto kNumStreams = 2;

cudaStream_t streams[kNumStreams];
bool stream_created = false;

// --------- Core ---------
void Init() {
  CudaWarmup<<<1, 1024>>>();
  DeviceSynchronize();
}

void DeviceSynchronize() { HANDLE_ERROR(cudaDeviceSynchronize()); }

void DeviceStreamSynchronize(const int stream_id) {
  HANDLE_ERROR(cudaStreamSynchronize(streams[stream_id]));
}

void AttachStreamMem(const int stream_id, void* addr) {
  if (!stream_created) {
    for (int i = 0; i < kNumStreams; i++) {
      HANDLE_ERROR(cudaStreamCreate(&streams[i]));
    }
    stream_created = true;
  }

  cudaStreamAttachMemAsync(streams[stream_id], addr);
}

// --------- Unified Memory ---------

void* UsmMalloc(std::size_t n) {
  void* tmp;
  HANDLE_ERROR(cudaMallocManaged(&tmp, n));
  std::cout << "accelerator::UsmMalloc() " << tmp << ": " << n << " bytes."
            << std::endl;
  return tmp;
}

void UsmFree(void* ptr) {
  std::cout << "accelerator::UsmFree() " << ptr << std::endl;
  if (ptr) {
    HANDLE_ERROR(cudaFree(ptr));
  }
}

// --------- CudaNn related ---------

void LaunchNnKenrnel(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* u_out,               /* stream base addr */
                     const Point4F* u_lnt_data,  /**/
                     const int max_leaf_size, const int stream_id) {
  const auto n_blocks = 1;
  constexpr auto n_threads = 1024;
  constexpr auto smem_size = 0;
  FindMinDistWarp6<<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
      u_lnt_data, u_q_points, u_leaf_indices, u_out, num_active_leafs,
      max_leaf_size);
}

}  // namespace redwood