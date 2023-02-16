#pragma once

#include "Redwood/Usm.hpp"
#include "Utils.hpp"

namespace rdc {

template <typename T>
struct NnBuffer {
  void Allocate(const int num_batch) {
    query_point.reserve(num_batch);
    leaf_idx.reserve(num_batch);
  }

  _NODISCARD const int* LData() const { return leaf_idx.data(); };
  _NODISCARD const T* QData() const { return query_point.data(); };
  _NODISCARD const size_t Size() const { return leaf_idx.size(); }

  void Clear() {
    // TODO: no need to clear every time, just overwrite the value
    leaf_idx.clear();
    query_point.clear();
  }

  void Push(const T q, const int leaf_id) {
    leaf_idx.push_back(leaf_id);
    query_point.push_back(q);
  }

  redwood::UsmVector<int> leaf_idx;
  redwood::UsmVector<T> query_point;
};

}  // namespace rdc