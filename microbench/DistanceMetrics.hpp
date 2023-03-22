#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "Point.hpp"

namespace dist {

struct Euclidean {
  float operator()(const Float4& p, const Float4& q) const {
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float dz = p.z - q.z;
    const float dw = p.w - q.w;
    return std::sqrt(dx * dx + dy * dy + dz * dz + dw * dw);
  }
};

struct Manhattan {
  float operator()(const Float4& p, const Float4& q) const {
    return std::abs(p.x - q.x) + std::abs(p.y - q.y) + std::abs(p.z - q.z) +
           std::abs(p.w - q.w);
  }
};

struct Chebyshev {
  float operator()(const Float4& p, const Float4& q) const {
    float dx = std::abs(p.x - q.x);
    float dy = std::abs(p.y - q.y);
    float dz = std::abs(p.z - q.z);
    float dw = std::abs(p.w - q.w);
    return std::max({dx, dy, dz, dw});
  }
};

struct EuclideanSquared {
  float operator()(const Float4& p, const Float4& q) const {
    const float dx = p.x - q.x;
    const float dy = p.y - q.y;
    const float dz = p.z - q.z;
    const float dw = p.w - q.w;
    return dx * dx + dy * dy + dz * dz + dw * dw;
  }
};

struct Minkowski {
  float p;

  Minkowski(const float p_value) : p(p_value) {
    if (p <= 0) {
      throw std::invalid_argument(
          "The p value for Minkowski  must be greater than 0.");
    }
  }

  float operator()(const Float4& p, const Float4& q) const {
    const float dx = std::abs(p.x - q.x);
    const float dy = std::abs(p.y - q.y);
    const float dz = std::abs(p.z - q.z);
    const float dw = std::abs(p.w - q.w);
    return std::pow(std::pow(dx, this->p) + std::pow(dy, this->p) +
                        std::pow(dz, this->p) + std::pow(dw, this->p),
                    1.0f / this->p);
  }
};

struct Canberra {
  float operator()(const Float4& p, const Float4& q) const {
    const float dx = std::abs(p.x - q.x) / (std::abs(p.x) + std::abs(q.x));
    const float dy = std::abs(p.y - q.y) / (std::abs(p.y) + std::abs(q.y));
    const float dz = std::abs(p.z - q.z) / (std::abs(p.z) + std::abs(q.z));
    const float dw = std::abs(p.w - q.w) / (std::abs(p.w) + std::abs(q.w));
    return dx + dy + dz + dw;
  }
};

struct BrayCurtis {
  float operator()(const Float4& p, const Float4& q) const {
    const float dx = std::abs(p.x - q.x);
    const float dy = std::abs(p.y - q.y);
    const float dz = std::abs(p.z - q.z);
    const float dw = std::abs(p.w - q.w);
    return (dx + dy + dz + dw) /
           (p.x + q.x + p.y + q.y + p.z + q.z + p.w + q.w);
  }
};

struct Cosine {
  float operator()(const Float4& p, const Float4& q) const {
    const float dot_product = p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
    const float p_norm =
        std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z + p.w * p.w);
    const float q_norm =
        std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);

    if (p_norm == 0.0f || q_norm == 0.0f) {
      throw std::domain_error("Cannot compute cosine  for zero-norm vectors.");
    }

    return 1.0f - (dot_product / (p_norm * q_norm));
  }
};
}  // namespace dist
