#pragma once

#include "../include/UsmAlloc.hpp"

namespace redwood {

// Single query
template <typename DataT, typename QueryT, typename ResultT>
struct BhBuffer {
  void Allocate(const int batch_size) {
    // If it grow larger then let it happen. I don't care
    leaf_nodes.reserve(batch_size);
    branch_data.reserve(batch_size);
  }

  size_t NumLeafsCollected() const { return leaf_nodes.size(); }
  size_t NumBranchCollected() const { return branch_data.size(); }

  void Clear() {
    // No need to clear the 'tem_result_br's, they will be overwrite
    tmp_count_br = 0;
    tmp_count_le = 0;
    leaf_nodes.clear();
    branch_data.clear();
  }

  // Getter/Setters
  void SetTask(const QueryT& q, const int q_idx) {
    my_query = q;
    my_q_idx = q_idx;
  }
  const int* LeafNodeData() const { return leaf_nodes.data(); };
  const DataT* BranchNodeData() const { return branch_data.data(); };

  void PushLeaf(const int leaf_id) { leaf_nodes.push_back(leaf_id); }
  void PushBranch(const DataT& com) { branch_data.push_back(com); }

  // Actual batch data , a single task with many many branch/leaf_idx
  QueryT my_query;
  int my_q_idx;

  redwood::UsmVector<int> leaf_nodes;
  redwood::UsmVector<DataT> branch_data;

  int tmp_count_br;
  int tmp_count_le;
};

template <typename T>
struct BhResult {
  BhResult(const int num_query) : results(num_query) {}

  UsmVector<T> results;
};

}  // namespace redwood