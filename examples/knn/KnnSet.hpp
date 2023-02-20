#pragma once

#include <algorithm>
#include <fstream>
#include <limits>

#include "../Utils.hpp"

template <typename T, int K>
struct KnnSet {
  // Insert a distance into the current result set, and matain the set is still
  // sorted. Shift everything after the inserted value to the back by one.
  void Insert(T value) {
    auto low = std::lower_bound(rank, rank + K, value);
    if (low != std::end(rank)) {
      // Shif the rest to the right
      for (int i = K - 2; i >= low - std::begin(rank); --i) {
        rank[i + 1] = rank[i];
      }
      *low = value;
    }
  }

  void Clear() { std::fill_n(rank, K, std::numeric_limits<float>::max()); }

  // Use this to get the least "Nearest" neighbor
  _NODISCARD T WorstDist() const { return rank[K - 1]; }

  void DebugPrint(std::ofstream& os) const {
    for (int i = 0; i < K; ++i) {
      os << i << ":\t" << rank[i] << '\n';
    }
  }

  void DebugPrint() const {
    for (int i = 0; i < K; ++i) {
      std::cout << i << ":\t" << rank[i] << '\n';
    }
  }

  // Assume sorted
  T rank[K];
};

// For Nearest Neighbor
template <typename T>
struct KnnSet<T, 1> {
  void Insert(T value) {
    if (value < dat) dat = value;
  }

  void Clear() { dat = std::numeric_limits<float>::max(); }

  _NODISCARD T WorstDist() const { return dat; }

  void DebugPrint(std::ofstream& os) const { os << dat << '\n'; }

  void DebugPrint() const { std::cout << dat << '\n'; }

  T dat;
};
