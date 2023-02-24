#pragma once

#include "Redwood.hpp"
#include "Utils.hpp"

namespace rdc {

// A Warpper class
//
// Potentially, make this part user's code. Because, the user will write Kernel
// code. And the user should provide buffer that can be passed to his kernel.
struct IndicesBuffer {
  void Allocate(const int batch_size) {
    // If it grow larger then let it happen. I don't care
    leaf_nodes.reserve(batch_size);
  }

  _NODISCARD size_t Size() const { return leaf_nodes.size(); }

  void Clear() {
    // TODO: No need to clear the 'tem_result_br's, they will be overwrite
    leaf_nodes.clear();
  }

  _NODISCARD const int* Data() const { return leaf_nodes.data(); }

  void PushLeaf(const int leaf_id) { leaf_nodes.push_back(leaf_id); }

  // Data
  redwood::UsmVector<int> leaf_nodes;
};

}  // namespace rdc