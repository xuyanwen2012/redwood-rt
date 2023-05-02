#include "CudaUtils.cuh"
#include "Redwood/Core.hpp"

namespace redwood {

std::vector<cudaStream_t> streams;
int stored_num_threads;

__global__ void CudaWarmup() {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

void Init(const int num_threads) {
  stored_num_threads = num_threads;

  streams.resize(num_threads * kNumStreams);
  for (int tid = 0; tid < num_threads; ++tid) {
    for (int stream_id = 0; stream_id < kNumStreams; stream_id++) {
      HANDLE_ERROR(cudaStreamCreate(&streams[tid * kNumStreams + stream_id]));
    }
  }

  CudaWarmup<<<1, 1024>>>();
  DeviceSynchronize();
}

void DeviceSynchronize() { HANDLE_ERROR(cudaDeviceSynchronize()); }

void DeviceStreamSynchronize(const int tid, const int stream_id) {
  HANDLE_ERROR(cudaStreamSynchronize(streams[tid * kNumStreams + stream_id]));
}

void AttachStreamMem(const int tid, const int stream_id, void* addr) {
  cudaStreamAttachMemAsync(streams[tid * kNumStreams + stream_id], addr);
}

}  // namespace redwood