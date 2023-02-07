#include <CL/sycl.hpp>
#include <iostream>

#include "SyclUtils.hpp"

extern sycl::device device;
extern sycl::context ctx;

namespace redwood::accelerator {

void* UsmMalloc(std::size_t n) {
  std::cout << "accelerator::UsmMalloc() " << n << std::endl;
  void* tmp;
  tmp = sycl::malloc_shared(n, device, ctx);
  return tmp;
}

void UsmFree(void* ptr) {
  std::cout << "accelerator::UsmFree() " << ptr << std::endl;
  if (ptr) {
    sycl::free(ptr, ctx);
  }
}

}  // namespace redwood::accelerator