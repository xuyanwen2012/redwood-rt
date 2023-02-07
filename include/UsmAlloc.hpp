#pragma once

#include <vector>

#include "accelerator/Usm.hpp"

namespace redwood {

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
    return static_cast<value_type*>(
        accelerator::UsmMalloc(n * sizeof(value_type)));
  }

  void deallocate(pointer p, std::size_t n) {
    if (p) {
      accelerator::UsmFree(p);
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