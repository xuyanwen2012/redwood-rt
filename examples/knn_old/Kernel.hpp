#pragma once

#include <cmath>

#include "Redwood/Point.hpp"

// Host version, need another advice implementation
inline float KernelFunc(const Point4F p, const Point4F q) {
  auto dist = float();

  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}
