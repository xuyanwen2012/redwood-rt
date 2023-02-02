#include <iostream>

// #include "../Backend.hpp"
#include "CudaUtils.cuh"
#include "cuda_runtime.h"

namespace redwood::internal {

void* UsmAlloc(std::size_t n) {
  std::cout << "internal::UsmAlloc() " << n << std::endl;
  void* tmp;
  HANDLE_ERROR(cudaMallocManaged(&tmp, n));
  return tmp;
}
void UsmDeAlloc(void* ptr) {
  std::cout << "internal::UsmDeAlloc() " << ptr << std::endl;
  if (ptr) {
    HANDLE_ERROR(cudaFree(ptr));
  }
}

}  // namespace redwood::internal