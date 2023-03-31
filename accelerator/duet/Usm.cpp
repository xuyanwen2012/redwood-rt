#include "Redwood/Usm.hpp"

#include <cstdlib>
#include <iostream>

constexpr auto kDuetAlignment = 64;

namespace redwood {

// 'USM'
void* UsmMalloc(std::size_t n) {
  std::cout << "std::aligned_alloc() " << n << std::endl;
  return aligned_alloc(kDuetAlignment, n);
}

void UsmFree(void* ptr) {
  std::cout << "std::free() " << ptr << std::endl;
  if (ptr) free(ptr);
}

}  // namespace redwood