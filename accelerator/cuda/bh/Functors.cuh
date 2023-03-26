#pragma once

#include "../../Point.hpp"

namespace dist_cuda {

inline __host__ __device__ float dist_sqr(const Point4F p, const Point4F q) {
  const auto dx = p.data[0] - q.data[0];
  const auto dy = p.data[1] - q.data[1];
  const auto dz = p.data[2] - q.data[2];
  return dx * dx + dy * dy + dz * dz + 1e-9f;
}

struct Gravity {
  __host__ __device__ float operator()(const Point4F p, const Point4F q) const {
    const auto dist_sqr = dist_sqr(p, q);
    const auto inv_dist = rsqrtf(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

struct Gaussian {
  __host__ __device__ float operator()(const Point4F p, const Point4F q) const {
    const auto dist_sqr = dist_sqr(p, q);
    const auto sigma_sqr = 0.1f * 0.1f;  // set sigma here
    const auto factor = 1.0f / (sqrtf(2.0f * M_PI) * sigma_sqr);
    const auto exponent = -dist_sqr / (2.0f * sigma_sqr);
    const auto with_mass = factor * expf(exponent) * p.data[3];
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

struct TopHat {
  __host__ __device__ float operator()(const Point4F p, const Point4F q) const {
    const auto dist_sqr = dist_sqr(p, q);
    constexpr auto r = 0.1f;  // set radius here
    constexpr auto factor = 3.0f / (4.0f * M_PI * r * r * r);
    const auto with_mass = factor * (dist_sqr <= r * r ? p.data[3] : 0.0f);
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

}  // namespace dist_cuda