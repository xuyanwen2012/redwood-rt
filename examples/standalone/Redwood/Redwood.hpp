#pragma once

#include <vector>

namespace redwood {

// --- Public APIs (need to be implemented by the backends) --
void Init();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

void AttachStreamMem(int stream_id, void* addr);

// CUDA Only
inline void AttachStream(int stream_id, void* addr) {
  // if constexpr (kRedwoodBackend == redwood::Backends::kCuda)
  AttachStreamMem(stream_id, addr);
}

// --- Unified Memory ---

void* UsmMalloc(std::size_t n);

template <typename T>
T* UsmMalloc(std::size_t n) {
  return static_cast<T*>(UsmMalloc(n * sizeof(T)));
}

void UsmFree(void* ptr);

template <typename T>
class UsmAlloc {
 public:
  // must not change name
  using value_type = T;
  using pointer = value_type*;

  UsmAlloc() noexcept = default;

  template <typename U>
  UsmAlloc(const UsmAlloc<U>&) noexcept {}

  // must not change name
  value_type* allocate(std::size_t n, const void* = nullptr) {
    return static_cast<value_type*>(UsmMalloc(n * sizeof(value_type)));
  }

  void deallocate(pointer p, std::size_t n) {
    if (p) {
      UsmFree(p);
    }
  }
};

/* Equality operators */
template <class T, class U>
bool operator==(const UsmAlloc<T>&, const UsmAlloc<U>&) {
  return true;
}

template <class T, class U>
bool operator!=(const UsmAlloc<T>&, const UsmAlloc<U>&) {
  return false;
}

template <typename T>
using UsmVector = std::vector<T, redwood::UsmAlloc<T>>;

}  // namespace redwood