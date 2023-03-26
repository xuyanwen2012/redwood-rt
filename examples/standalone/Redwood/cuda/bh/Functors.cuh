#pragma once

#include "../../Point.hpp"

namespace dist_cuda {

struct Gravity {
  __host__ __device__ float operator()(const Point4F& p,
                                       const Point4F& q) const {
    const auto dx = p.data[0] - q.data[0];
    const auto dy = p.data[1] - q.data[1];
    const auto dz = p.data[2] - q.data[2];
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
    const auto inv_dist = rsqrtf(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

}  // namespace dist_cuda