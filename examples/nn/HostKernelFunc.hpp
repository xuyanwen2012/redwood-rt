#pragma once

#include <cmath>

#include "PointCloud.hpp"

struct MyFunctorHost {
  inline float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += diff * diff;
    }

    return sqrtf(dist);
  }
};
