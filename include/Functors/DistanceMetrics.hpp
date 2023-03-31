#pragma once

#include <cmath>

#include "Redwood/Point.hpp"

#ifdef __CUDACC__
#define _REDWOOD_KERNEL __host__ __device__
#else
#define _REDWOOD_KERNEL
#endif

#ifdef __CUDACC__
#define MAX(x, y) fmaxf(x, y)
#define SQRTF(x) sqrtf(x)
#define ABS(x) abs(x)
#else
#define MAX(x, y) std::max(x, y)
#define SQRTF(x) std::sqrt(x)
#define ABS(x) std::abs(x)
#endif

#define X data[0]
#define Y data[1]
#define Z data[2]
#define W data[3]

namespace dist {
struct Euclidean {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    const auto dx = p.X - q.X;
    const auto dy = p.Y - q.Y;
    const auto dz = p.Z - q.Z;
    const auto dw = p.W - q.W;
    return SQRTF(dx * dx + dy * dy + dz * dz + dw * dw);
  }

  _REDWOOD_KERNEL float operator()(const float a, const float b) const {
    const auto dx = a - b;
    return SQRTF(dx * dx);
  }
};

struct Manhattan {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    return ABS(p.X - q.X) + ABS(p.Y - q.Y) + ABS(p.Z - q.Z) + ABS(p.W - q.W);
  }
};

struct Chebyshev {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    const auto dx = ABS(p.X - q.X);
    const auto dy = ABS(p.Y - q.Y);
    const auto dz = ABS(p.Z - q.Z);
    const auto dw = ABS(p.W - q.W);
    auto tmp1 = MAX(dx, dy);
    auto tmp2 = MAX(dz, dw);
    return MAX(tmp1, tmp2);
  }
};

}  // namespace dist
