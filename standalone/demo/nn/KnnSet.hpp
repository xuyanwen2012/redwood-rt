#pragma once

#include <limits>

#include "../Utils.hpp"

// Specialization For NN
struct KnnSet {
  void Insert(const float value) {
    if (value < dat) dat = value;
  }

  void Reset() { dat = std::numeric_limits<float>::max(); }

  _NODISCARD float WorstDist() const { return dat; }

  float dat;
};