#pragma once

#include <cstdlib>

namespace redwood::accelerator {

void* UsmMalloc(std::size_t n);
void UsmFree(void* ptr);

}  // namespace redwood::accelerator
