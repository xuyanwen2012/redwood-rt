#pragma once

#include <cmath>

#include "Redwood/Point.hpp"

namespace dist {
struct Euclidean {
  float operator()(const Point4F& p, const Point4F& q) const {
    const float dx = p.data[0] - q.data[0];
    const float dy = p.data[1] - q.data[1];
    const float dz = p.data[2] - q.data[2];
    const float dw = p.data[3] - q.data[3];
    return std::sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
  }

  float operator()(const float a, const float b) const {
    const float dx = a - b;
    return std::sqrt(dx * dx);
  }
};
}  // namespace dist
