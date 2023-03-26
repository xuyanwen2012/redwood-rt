#pragma once

#include "../../Point.hpp"

namespace dist_cuda {
struct Euclidean {
  __host__ __device__ float operator()(const Point4F& p,
                                       const Point4F& q) const {
    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dw = p.data[3] - q.data[3];
    return sqrtf(dx * dx + dy * dy + dz * dz + dw * dw);
  }
};

struct Manhattan {
  __host__ __device__ float operator()(const Point4F p, const Point4F q) const {
    return abs(p.data[0] - q.data[0]) + abs(p.data[1] - q.data[1]) +
           abs(p.data[2] - q.data[2]) + abs(p.data[3] - q.data[3]);
  }
};

struct Chebyshev {
  __host__ __device__ float operator()(const Point4F p, const Point4F q) const {
    const float dx = abs(p.data[0] - q.data[0]);
    const float dy = abs(p.data[1] - q.data[1]);
    const float dz = abs(p.data[2] - q.data[2]);
    const float dw = abs(p.data[3] - q.data[3]);
    return fmaxf(fmaxf(fmaxf(dx, dy), dz), dw);
  }
};

}  // namespace dist_cuda