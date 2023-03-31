#include "Redwood/Usm.hpp"

#include <CL/sycl.hpp>
#include <iostream>

namespace redwood {

extern sycl::device device;
extern sycl::context ctx;

void* UsmMalloc(const std::size_t n) {
  void* tmp;
  std::cout << "accelerator::UsmMalloc() " << tmp << ": " << n << " bytes."
            << std::endl;
  tmp = sycl::malloc_shared(n, device, ctx);
  return tmp;
}

void UsmFree(void* ptr) {
  std::cout << "accelerator::UsmFree() " << ptr << std::endl;
  if (ptr) {
    sycl::free(ptr, ctx);
  }
}

}  // namespace redwood