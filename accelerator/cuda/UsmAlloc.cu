#include <iostream>

#include "CudaUtils.cuh"
#include "cuda_runtime.h"

namespace redwood::accelerator {

void* UsmMalloc(std::size_t n) {
  std::cout << "accelerator::UsmMalloc() " << n << std::endl;
  void* tmp;
  HANDLE_ERROR(cudaMallocManaged(&tmp, n));
  return tmp;
}
void UsmFree(void* ptr) {
  std::cout << "accelerator::UsmFree() " << ptr << std::endl;
  if (ptr) {
    HANDLE_ERROR(cudaFree(ptr));
  }
}

}  // namespace redwood::accelerator