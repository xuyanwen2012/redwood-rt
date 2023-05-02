#pragma once

#include <cmath>

#include "Redwood/Point.hpp"

#ifdef __CUDACC__
#define _REDWOOD_KERNEL __host__ __device__
#define _REDWOOD_KERNEL_INLINE __host__ __device__ __forceinline__
#else
#define _REDWOOD_KERNEL
#define _REDWOOD_KERNEL_INLINE
#endif  // #ifdef __CUDACC__

#ifdef __CUDACC__
#define MAX(x, y) fmaxf(x, y)
#define SQRTF(x) sqrtf(x)
#define ABS(x) abs(x)
#else
#define MAX(x, y) std::max(x, y)
#define SQRTF(x) std::sqrt(x)
#define ABS(x) std::abs(x)
#endif  // #ifdef __CUDACC__

#ifndef X
#define X data[0]
#define Y data[1]
#define Z data[2]
#define W data[3]
#define SOFTENING 1e-9f
#endif  // #ifndef X

namespace dist {
struct Euclidean {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    const auto dx = p.X - q.X;
    const auto dy = p.Y - q.Y;
    const auto dz = p.Z - q.Z;
    const auto dw = p.W - q.W;
    return SQRTF(dx * dx + dy * dy + dz * dz + dw * dw + SOFTENING);
  }

  _REDWOOD_KERNEL float operator()(const float a, const float b) const {
    const auto dx = a - b;
    return SQRTF(dx * dx + SOFTENING);
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

struct Gravity {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    const auto dx = p.X - q.X;
    const auto dy = p.Y - q.Y;
    const auto dz = p.Z - q.Z;
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    const auto inv_dist = 1.0f / SQRTF(dist_sqr);
    const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
    const auto with_mass = inv_dist3 * p.data[3];
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

struct Gaussian {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    const auto dx = p.X - q.X;
    const auto dy = p.Y - q.Y;
    const auto dz = p.Z - q.Z;
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    constexpr auto sigma_sqr = 0.1f * 0.1f;
    const auto factor = 1.0f / (SQRTF(2.0f * M_PI) * sigma_sqr);
    const auto exponent = -dist_sqr / (2.0f * sigma_sqr);
    const auto with_mass = factor * expf(exponent) * p.W;
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

struct TopHat {
  _REDWOOD_KERNEL float operator()(const Point4F p, const Point4F q) const {
    const auto dx = p.X - q.X;
    const auto dy = p.Y - q.Y;
    const auto dz = p.Z - q.Z;
    const auto dist_sqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    constexpr auto r = 0.1f;
    constexpr auto factor = 3.0f / (4.0f * M_PI * r * r * r);
    const auto with_mass = factor * (dist_sqr <= r * r ? p.W : 0.0f);
    return dx * with_mass + dy * with_mass + dz * with_mass;
  }
};

}  // namespace dist
