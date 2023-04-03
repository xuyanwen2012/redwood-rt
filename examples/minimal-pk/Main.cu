
#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cuda_runtime.h"

constexpr auto kNumBlocks = 1;
constexpr auto kNumThreads = 1024;

namespace cg = cooperative_groups;

// Other Utils
static void handle_error(const cudaError_t err, const char* file,
                         const int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

inline bool cudaAllocMapped(void** cpuPtr, void** gpuPtr, const size_t size) {
  if (!cpuPtr || !gpuPtr || size == 0) return false;

  HANDLE_ERROR(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped));
  HANDLE_ERROR(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0));

  memset(*cpuPtr, 0, size);

  printf("cudaAllocMapped %zu bytes, CPU %p GPU %p\n", size, *cpuPtr, *gpuPtr);
  return true;
}

inline bool cudaAllocMapped(void** ptr, const size_t size) {
  void* cpuPtr = nullptr;
  void* gpuPtr = nullptr;

  if (!ptr || size == 0) return false;

  if (!cudaAllocMapped(&cpuPtr, &gpuPtr, size)) return false;

  if (cpuPtr != gpuPtr) {
    printf(
        "cudaAllocMapped() - addresses of CPU and GPU pointers don't match\n");
    return false;
  }

  *ptr = gpuPtr;
  return true;
}

template <typename T>
inline bool cudaAllocMapped(T** ptr, const size_t size) {
  return cudaAllocMapped((void**)ptr, size);
}

__device__ __forceinline__ void WaitCPU(volatile int* com) {
  int block_id = blockIdx.x;
  while (com[block_id] != 1 && com[kNumBlocks] != 1) {
    __threadfence_system();
  }
}

__device__ __forceinline__ void WorkComplete(volatile int* com) {
  int block_id = blockIdx.x;
  com[block_id] = 0;
}

__global__ void PersistentKernel(volatile int* com) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  while (com[kNumBlocks] != 1) {
    if (tid == 0) WaitCPU(com);
    __syncthreads();

    // cancelling point
    if (com[kNumBlocks] == 1) return;

    // --- GPU do something here ---
    // -----

    if (tid == 0) WorkComplete(com);
  }
}

void StartGPU(int* com) {
  // atomic?
  for (int i = 0; i < kNumBlocks; ++i) com[i] = 1;
}

void WaitGPU(int* com) {
  int sum;
  do {
    sum = 0;
    asm volatile("" ::: "memory");
    for (int i = 0; i < kNumBlocks; ++i) sum |= com[i];
  } while (sum != 0);
}

void EndGPU(int* com) {
  printf("cpu is ending GPU\n");
  com[kNumBlocks] = 1;
}

int main(int argc, char** argv) {
  // Allocate 2 integer in CPU-GPU shared memory
  int* u_com = nullptr;
  cudaAllocMapped(&u_com, sizeof(int) * (kNumBlocks + 1));

  // Launching the PK only once
  PersistentKernel<<<kNumBlocks, kNumThreads>>>(u_com);

  // CPU sending signal to GPU
  StartGPU(u_com);

  // CPU spin, waiting for GPU finish
  WaitGPU(u_com);

  // Tell the GPU to shuttdown
  EndGPU(u_com);

  HANDLE_ERROR(cudaFreeHost(u_com));

  return 0;
}