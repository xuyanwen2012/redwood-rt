#pragma once

#include <limits>

#include "../include/UsmAlloc.hpp"

namespace redwood {

template <typename T>
struct NnBuffer {
  void Allocate(const int num_batch) {
    query_point.reserve(num_batch);
    query_idx.reserve(num_batch);
    leaf_idx.reserve(num_batch);
  }

  size_t Size() const { return leaf_idx.size(); }

  void Clear() {
    // TODO: no need to clear every time, just overwrite the value
    query_point.clear();
    query_idx.clear();
    leaf_idx.clear();
  }

  void Push(const T& q, const int q_idx, const int leaf_id) {
    query_point.push_back(q);
    query_idx.push_back(q_idx);
    leaf_idx.push_back(leaf_id);
  }

  UsmVector<T> query_point;
  UsmVector<int> query_idx;
  UsmVector<int> leaf_idx;
};

template <typename T>
struct NnResult {
  NnResult(const int num_query) : results(num_query) {
    std::fill(results.begin(), results.end(), std::numeric_limits<T>::max());
  }

  UsmVector<T> results;
};

}  // namespace redwood