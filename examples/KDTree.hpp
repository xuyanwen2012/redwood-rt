#pragma once

#include <cassert>
#include <numeric>
#include <vector>

#include "PointCloud.hpp"
#include "Utils.hpp"

namespace kdt {

enum class Dir { kLeft = 0, kRight };

inline Dir FlipDir(const Dir dir) {
  return dir == Dir::kLeft ? Dir::kRight : Dir::kLeft;
}

struct Node {
  _NODISCARD bool IsLeaf() const {
    return left_child == nullptr && right_child == nullptr;
  }

  _NODISCARD Node *GetChild(const Dir dir) const {
    return dir == Dir::kLeft ? left_child : right_child;
  }

  union {
    struct {
      // Indices of points in leaf node, it basically specify a
      // range in the original dataset
      int idx_left;
      int idx_right;
    } leaf;

    struct {
      // Dimension used for subdivision. (e.g. 0, 1, 2)
      int axis;
      int idx_mid;
    } tree;
  } node_type;

  Node *left_child;
  Node *right_child;
  int uid;  // In this version this is used only for leaf nodes.
};

struct KdtParams {
  explicit KdtParams(const size_t leaf_size = 32) : leaf_max_size(leaf_size) {
    if (leaf_size == 0) {
      throw std::runtime_error("Error: 'leaf_size' must be above zero. ");
    }
  }

  size_t leaf_max_size;
};

struct KdtStatistic {
  // Tree building related statistics
  size_t num_leaf_nodes = 0;
  size_t num_branch_nodes = 0;
  int max_depth = 0;

  // Tree Traversal related statistics
  int leaf_node_visited = 0;
  int branch_node_visited = 0;
  int total_elements_reduced = 0;
};

class KdTree {
  using T = Point4F;

 public:
  KdTree() = delete;

  explicit KdTree(const KdtParams params, const T *in_data, const int n)
      : root_(), in_data_ref_(in_data), node_table_size_(), params_(params) {
    BuildTree(n);
  }

  void BuildTree(const int size) {
    v_acc_.resize(size);
    std::iota(v_acc_.begin(), v_acc_.end(), 0u);

    root_ = BuildRecursive(0u, static_cast<int>(size) - 1, 0);

    // New Stuff
    // Create Node content table on host
    node_table_size_ = statistic_.num_leaf_nodes * params_.leaf_max_size;

    // h_node_content_table_ = DataSetT(node_table_size_);
    // LoadPayloadRecursive(root_);

    if constexpr (constexpr auto print = true) {
      std::cout << "Tree Statistic: \n"
                << "\tNum leaf nodes: \t" << statistic_.num_leaf_nodes << '\n'
                << "\tNum branch nodes: \t" << statistic_.num_branch_nodes
                << '\n'
                << '\n'
                << "\tMax Depth: \t" << statistic_.max_depth << '\n'
                << std::endl;
    }
  }

  void LoadPayload(T *usm_leaf_node_table) {
    assert(usm_leaf_node_table != nullptr);
    LoadPayloadRecursive(root_, usm_leaf_node_table);
  }

  _NODISCARD KdtStatistic GetStats() const { return statistic_; }
  _NODISCARD KdtParams GetParams() const { return params_; }
  //   _NODISCARD DataSetT& GetNodeContentTable() { return
  //   h_node_content_table_; }
  _NODISCARD const Node *GetRoot() const { return root_; }

  Node *BuildRecursive(const int left_idx, const int right_idx,
                       const int depth) {
    const auto node = new Node;

    if (right_idx - left_idx <= (int)params_.leaf_max_size)  // minimum is 1
    {
      ++statistic_.num_leaf_nodes;
      statistic_.max_depth = std::max(depth, statistic_.max_depth);

      // Build as leaf node
      node->node_type.leaf.idx_left = left_idx;
      node->node_type.leaf.idx_right = right_idx;
      node->left_child = nullptr;
      node->right_child = nullptr;
      node->uid = GetNextId();
    } else {
      ++statistic_.num_branch_nodes;

      // Build as tree node
      const auto axis = depth % 4;
      const auto mid_idx = (left_idx + right_idx) / 2;

      // I am splitting at the median
      std::nth_element(
          v_acc_.begin() + left_idx, v_acc_.begin() + mid_idx,
          v_acc_.begin() + right_idx + 1, [&](const auto lhs, const auto rhs) {
            return in_data_ref_[lhs].data[axis] < in_data_ref_[rhs].data[axis];
          });

      // Mid point as the node, then everything on the left will
      // be in left child, everything on the right in the right
      // child.
      node->node_type.tree.axis = axis;
      node->node_type.tree.idx_mid = mid_idx;
      node->left_child = BuildRecursive(left_idx, mid_idx - 1, depth + 1);
      node->right_child = BuildRecursive(mid_idx + 1, right_idx, depth + 1);
      node->uid = -1;
    }

    return node;
  }

  void LoadPayloadRecursive(const Node *cur, T *usm_leaf_node_table) {
    if (cur->IsLeaf()) {
      auto counter = 0u;
      const auto offset = cur->uid * params_.leaf_max_size;

      for (auto i = cur->node_type.leaf.idx_left;
           i <= cur->node_type.leaf.idx_right; ++i) {
        const auto idx = v_acc_[i];
        // std::cout << offset + counter << std::endl;
        usm_leaf_node_table[offset + counter] = in_data_ref_[idx];
        ++counter;
      }

      // Making sure remaining are filled.
      while (counter < params_.leaf_max_size) {
        usm_leaf_node_table[offset + counter].data[0] =
            std::numeric_limits<float>::max();
        ++counter;
      }
    } else {
      LoadPayloadRecursive(cur->left_child, usm_leaf_node_table);
      LoadPayloadRecursive(cur->right_child, usm_leaf_node_table);
    }
  }

  static int GetNextId() {
    static int uid_counter = 0;
    return uid_counter++;
  }

  // Accessor
  Node *root_;
  std::vector<int> v_acc_;

  // Datasets (ref to Input Data, and the Node Contents)
  //   const DataSetT& data_set_;
  const T *in_data_ref_;

  //   DataSetT h_node_content_table_;
  int node_table_size_;

  // Statistics informations for/of the tree construction
  KdtParams params_;
  KdtStatistic statistic_;
};

}  // namespace kdt
