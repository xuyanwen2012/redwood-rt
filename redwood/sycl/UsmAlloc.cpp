#include <CL/sycl.hpp>
#include <iostream>

#include "SyclUtils.hpp"

constexpr auto kNumStreams = 2;

// need to be externed?
sycl::device device;
sycl::context ctx;
sycl::queue qs[kNumStreams];

namespace redwood::internal {

void InitUsm() {
  std::cout << "internal::InitUsm()" << std::endl;

  try {
    device = sycl::device(sycl::gpu_selector_v);
  } catch (const sycl::exception& e) {
    std::cout << "Cannot select a GPU\n" << e.what() << "\n";
    exit(1);
  }

  qs[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++)
    qs[i] = sycl::queue(qs[0].get_context(), device);

  ShowDevice(qs[0]);

  // WarmUp(qs[0]);
}

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