#pragma once

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>

#include "../Utils.hpp"
#include "Redwood/Point.hpp"

namespace kdt {
using Task = std::pair<int, Point4F>;

const int MAX_NODES = 1024;
enum class Dir { kLeft = 0, kRight };

inline Dir FlipDir(const Dir dir) {
  return dir == Dir::kLeft ? Dir::kRight : Dir::kLeft;
}


struct Node{
    _NODISCARD bool IsLeaf() const {
    return left_child == -1 && right_child == -1;
  }
  int left_child;
  int right_child;
  int axis;
  int uid;
  
};

struct Range{
  int low;
  int high;
};

/*
struct Node {
  _NODISCARD bool IsLeaf() const {
    return left_child == nullptr && right_child == nullptr;
  }

  _NODISCARD Node* GetChild(const Dir dir) const {
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
      int min_idx;
      int max_idx;
    } tree;
  } node_type;

  Node* left_child;
  Node* right_child;
  int uid;  // In this version this is used only for leaf nodes.
};
*/


struct KdtParams {
  explicit KdtParams(const int leaf_size = 32) : leaf_max_size(leaf_size) {
    if (leaf_size == 0) {
      throw std::runtime_error("Error: 'leaf_size' must be above zero. ");
    }
  }

  int leaf_max_size;
};

struct KdtStatistic {
  // Tree building related statistics
  int num_leaf_nodes = 0;
  int num_branch_nodes = 0;
  int max_depth = 0;

  // Tree Traversal related statistics
  int leaf_node_visited = 0;
  int branch_node_visited = 0;
  int total_elements_reduced = 0;
};

int next_node = 0;
Node nodes[MAX_NODES];
Range ranges[MAX_NODES];

int new_node(){
  int cur = next_node;
  nodes[cur].left_child = -1;
  nodes[cur].right_child = -1;
  nodes[cur].axis = -1;
  ranges[cur].low = -1;
  ranges[cur].high = -1;
  next_node += 1;
  return cur;
}

class KdTree {
  using T = Point4F;

 public:
  KdTree() = delete;

  explicit KdTree(const KdtParams params, T* in_data, const int n)
      : root_(), in_data_ref_(in_data), params_(params) {
    BuildTree(n);
  }

  void BuildTree(const int size) {

    root_ = BuildRecursive(0u, static_cast<int>(size) - 1, 0);

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

  void LoadPayload(T* usm_leaf_node_table) {
    assert(usm_leaf_node_table != nullptr);
    LoadPayloadRecursive(root_, usm_leaf_node_table);
  }

  _NODISCARD KdtStatistic GetStats() const { return statistic_; }
  _NODISCARD KdtParams GetParams() const { return params_; }
  _NODISCARD const int GetRoot() const { return root_; }

  int BuildRecursive(const int low, const int high,
                       const int depth) {
    int cur = new_node();

    if (high - low <= params_.leaf_max_size)  // minimum is 1
    {
      ++statistic_.num_leaf_nodes;
      statistic_.max_depth = std::max(depth, statistic_.max_depth);

      // Build as leaf node
      ranges[cur].low = low;
      ranges[cur].high = high;
      nodes[cur].uid = GetNextId();
    } else {
      ++statistic_.num_branch_nodes;

      // Build as tree node
      const auto axis = depth % 4;
      const auto mid_idx = (low + high) / 2;

      // I am splitting at the median
      std::nth_element(
          in_data_ref_+low,in_data_ref_+ mid_idx,
          in_data_ref_ + high + 1);

      // Mid point as the node, then everything on the left will
      // be in left child, everything on the right in the right
      // child.
      nodes[cur].axis = axis;
      
      nodes[cur].left_child = BuildRecursive(low, mid_idx - 1, depth + 1);
      nodes[cur].right_child = BuildRecursive(mid_idx + 1, high, depth + 1);
      // if left child is leaf node

      ranges[cur].low = ranges[nodes[cur].left_child].low;
      ranges[cur].high = ranges[nodes[cur].right_child].high;
      nodes[cur].uid = -1;
    }

    return cur;
  }

  void LoadPayloadRecursive(const int cur, T* usm_leaf_node_table) {
    if (nodes[cur].IsLeaf()) {
      auto counter = 0;
      const auto offset = nodes[cur].uid * params_.leaf_max_size;

      for (auto i = ranges[cur].low;
           i <= ranges[cur].high; ++i) {
        usm_leaf_node_table[offset + counter] = in_data_ref_[i];
        ++counter;
      }

      // Making sure remaining are filled.
      while (counter < params_.leaf_max_size) {
        usm_leaf_node_table[offset + counter].data[0] =
            std::numeric_limits<float>::max();
        ++counter;
      }
    } else {
      LoadPayloadRecursive(nodes[cur].left_child, usm_leaf_node_table);
      LoadPayloadRecursive(nodes[cur].right_child, usm_leaf_node_table);
    }
  }

  static int GetNextId() {
    static int uid_counter = 0;
    return uid_counter++;
  }

  // Accessor
  int root_;
  //std::vector<int> v_acc_;

  // Datasets (ref to Input Data, and the Node Contents)
  // Note: dangerous, do not use after LoadPayload
  T* in_data_ref_;

  // Statistics informations for/of the tree construction
  KdtParams params_;
  KdtStatistic statistic_;
};

inline std::tuple<int, int> GetSubRange(const Node* node, Task target) {
  if (!node->IsLeaf()) {
    if (target.first < node->node_type.tree.min_idx ||
        target.first > node->node_type.tree.max_idx) {
      return std::make_tuple(-1, -1);
    }
    // target  is completely contained within this node's range
    if (node->node_type.tree.min_idx <= target.first &&
        node->node_type.tree.max_idx >= target.first) {
      return std::make_tuple(node->node_type.tree.min_idx,
                             node->node_type.tree.max_idx);
    }
    /*
    // target is partially contained within this node's range
    std::tuple<int, int> left_range = GetSubRange(node->left_child, target);
    std::tuple<int, int> right_range = GetSubRange(node->right_child, target);

    if (std::get<0>(left_range) == -1 && std::get<1>(left_range) == -1){
      return right_range;
    }else if (std::get<0>(right_range)  == -1 &&std::get<1>(right_range)  ==
    -1){ return left_range; }else{ return
    std::make_tuple(std::min(std::get<0>(left_range) , std::get<0>(right_range)
    ), std::max(std::get<1>(left_range), std::get<1>(right_range)));
    }
    */
    return std::make_tuple(-1, -1);

  } else {
    return std::make_tuple(-1, -1);
  }
}
}  // namespace kdt
