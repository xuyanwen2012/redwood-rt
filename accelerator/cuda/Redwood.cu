#include "CudaUtils.cuh"
#include "Redwood/Core.hpp"
#include "UserKernels.cuh"
#include "cuda_runtime.h"

// ---------------------- Global variables ----------------

namespace redwood {

cudaStream_t streams[kNumStreams];
bool stream_created = false;

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

void ComputeOneBatchAsync(const int* u_leaf_indices,  /**/
                          const int num_active_leafs, /**/
                          float* out,                 /**/
                          const Point4F* u_lnt_data,  /**/
                          const int* u_lnt_sizes,     /**/
                          const Point4F q,            /**/
                          const int stream_id) {
  const auto n_blocks = 1;
  constexpr auto n_threads = 1024;
  constexpr auto smem_size = 0;

  // CudaBarnesHutDebug<64><<<n_blocks, 1, smem_size, streams[stream_id]>>>(
  //     u_leaf_indices, u_lnt_data, u_lnt_sizes, q, out, num_active_leafs);
  CudaBarnesHut<64><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
      u_leaf_indices, u_lnt_data, u_lnt_sizes, q, out, num_active_leafs);
}

void ProcessKnnAsync(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* out,                 /**/
                     const Point4F* u_lnt_data,  /**/
                     const int* u_lnt_sizes,     /**/
                     const int stream_id) {
  const auto n_blocks = 1;
  constexpr auto n_threads = 1024;
  constexpr auto smem_size = 0;

  CudaKnnDebug<64><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
      u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
      u_lnt_sizes);
}

void ProcessNnAsync(const int* u_leaf_indices,  /**/
                    const Point4F* u_q_points,  /**/
                    const int num_active_leafs, /**/
                    float* out,                 /**/
                    const Point4F* u_lnt_data,  /**/
                    const int* u_lnt_sizes,     /**/
                    const int max_leaf_size,    /**/
                    const int stream_id) {
  const auto n_blocks = 1;
  constexpr auto n_threads = 512;
  constexpr auto smem_size = 0;

  using Kernel = FindMinDistWarp6;

  switch (max_leaf_size) {
    case 1024:
      Kernel<1024><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
          u_lnt_sizes);
      break;
    case 512:
      Kernel<512><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
          u_lnt_sizes);
      break;
    case 256:
      Kernel<256><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
          u_lnt_sizes);
      break;
    case 128:
      Kernel<128><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
          u_lnt_sizes);
      break;
    case 64:
      Kernel<64><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
          u_lnt_sizes);
      break;
    case 32:
      Kernel<32><<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
          u_leaf_indices, u_q_points, num_active_leafs, out, u_lnt_data,
          u_lnt_sizes);
      break;
    default:
      std::cout << "Should not happen." << std::endl;
      exit(0);
  };
}

}  // namespace redwood