#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"

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
