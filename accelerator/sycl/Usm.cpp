#include "Redwood/Usm.hpp"

#include <sycl/sycl.hpp>

extern sycl::device g_device;
extern sycl::context g_context;

namespace redwood {

// USM is SYCL shared memory: accessible from both host and device, so the same
// pointer the kernels use can also be read/written directly on the host (as the
// CPU traversal does). Created against the backend context from Core.cpp.
void* UsmMalloc(std::size_t n) {
  if (n == 0) return nullptr;
  return sycl::malloc_shared(n, g_device, g_context);
}

void UsmFree(void* ptr) {
  if (ptr) sycl::free(ptr, g_context);
}

}  // namespace redwood
