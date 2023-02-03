#include <CL/sycl.hpp>
#include <iostream>

#include "SyclUtils.hpp"

namespace redwood::internal {

void* UsmAlloc(std::size_t n) {
  std::cout << "internal::UsmAlloc() " << n << std::endl;
  void* tmp;
  tmp = sycl::malloc_shared(n, device, ctx);
  return tmp;
}
void UsmDeAlloc(void* ptr) {
  std::cout << "internal::UsmDeAlloc() " << ptr << std::endl;
  if (ptr) {
    sycl::free(ptr, ctx);
  }
}

}  // namespace redwood::internal