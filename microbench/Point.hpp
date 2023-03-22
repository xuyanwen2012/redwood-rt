#pragma once

#include <random>
#include <vector>

struct Float4 {
  float x, y, z, w;
};

std::vector<Float4> GenerateRandomFloat4(const size_t num_points) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-1000.0, 1000.0);

  std::vector<Float4> points(num_points);
  for (auto& point : points) {
    point = {dis(gen), dis(gen), dis(gen), dis(gen)};
  }
  return points;
}
