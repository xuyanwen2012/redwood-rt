#include <iostream>

#include "Usm.hpp"

namespace redwood {

void* UsmMalloc(std::size_t n) {
  std::cout << "std::malloc() " << n << std::endl;
  return malloc(n);
}

void UsmFree(void* ptr) {
  std::cout << "std::free() " << ptr << std::endl;
  if (ptr) {
    free(ptr);
  }
}

}  // namespace redwood