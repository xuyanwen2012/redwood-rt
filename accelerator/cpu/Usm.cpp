#include "Redwood/Usm.hpp"

#include <cstdlib>

namespace redwood {

// On a CPU "backend", Unified Shared Memory is just regular host memory.
// We over-align to a cache line so the layout matches the other backends
// (CUDA managed memory and Duet's aligned_alloc both hand back aligned
// pointers), which keeps cross-backend behaviour identical in tests.
constexpr std::size_t kCpuAlignment = 64;

void* UsmMalloc(std::size_t n) {
  if (n == 0) return nullptr;
  // aligned_alloc requires the size to be a multiple of the alignment.
  const std::size_t rounded = (n + kCpuAlignment - 1) & ~(kCpuAlignment - 1);
  return std::aligned_alloc(kCpuAlignment, rounded);
}

void UsmFree(void* ptr) {
  if (ptr) std::free(ptr);
}

}  // namespace redwood
