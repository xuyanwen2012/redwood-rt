#include <iostream>
#include <limits>

#include "../Utils.hpp"

template <typename T, int K>
struct KnnSet {
  // Insert a distance into the current result set, and matain the set is still
  // sorted. Shift everything after the inserted value to the back by one.
  void Insert(const T value) {
    auto low = std::lower_bound(rank, rank + K, value);
    if (low != std::end(rank)) {
      // Shif the rest to the right
      for (int i = K - 2; i >= low - std::begin(rank); --i) {
        rank[i + 1] = rank[i];
      }
      *low = value;
    }
  }

  void Reset() { std::fill_n(rank, K, std::numeric_limits<T>::max()); }

  // Use this to get the least "Nearest" neighbor
  _NODISCARD T WorstDist() const { return rank[K - 1]; }

  void DebugPrint() const {
    for (int i = 0; i < K; ++i) {
      std::cout << i << ":\t" << rank[i] << '\n';
    }
  }

  // Assume sorted
 private:
  T rank[K];
};

// Specialization For NN
template <typename T>
struct KnnSet<T, 1> {
  void Insert(const T value) {
    if (value < dat) dat = value;
  }

  void Reset() { dat = std::numeric_limits<T>::max(); }

  _NODISCARD T WorstDist() const { return dat; }

 private:
  T dat;
};