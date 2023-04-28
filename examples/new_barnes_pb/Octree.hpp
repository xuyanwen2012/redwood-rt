#pragma once

#include <array>
#include <numeric>
#include <vector>

#include "Redwood/Point.hpp"

namespace oct {
#define X 0
#define Y 1
#define Z 2
#define MASS 3

using IndexT = int;
template <typename T>
struct DataRange{
  int start;
  int end;
}
// For oct tree the bounding box is always 3D
template <typename T>
struct BoundingBox {
  Point<3, T> dimension;
  Point<3, T> center;

  static BoundingBox Unit() {
    return {
        Point<3, T>{1.0, 1.0, 1.0},  // dimension
        Point<3, T>{0.5, 0.5, 0.5}   // center position
    };
  }
};

template <typename T>
struct OctreeParams {
  explicit OctreeParams(const T theta, const size_t leaf_size,
                        const BoundingBox<T> box = BoundingBox<T>::Unit())
      : theta_val(theta), leaf_max_size(leaf_size), starting_box(box) {
    if (leaf_size == 0) {
      throw std::runtime_error("Error: 'leaf_size' must be above zero. ");
    }
  }

  T theta_val;
  size_t leaf_max_size;
  BoundingBox<T> starting_box;
};

struct OctreeStatistic {
  // Tree building related statistics
  int num_leaf_nodes = 0;
  int num_branch_nodes = 0;
  int max_depth = 0;
};

template <typename T>
struct Node {
  using PointT = Point<4, T>;

  Node()
      : children(),
        bounding_box(),
        is_leaf(true),
        node_mass(),
        node_weighted_pos(),
        uid(-1) {
    return;
  }

  bool IsLeaf() const { return is_leaf; }

  const PointT CenterOfMass() const { return center_of_mass; }

  std::array<Node*, 8> children;
  std::vector<IndexT> bodies;
  BoundingBox<T> bounding_box;
  bool is_leaf;
  T node_mass;
  PointT node_weighted_pos;  // the MASS field is not used.
  int uid;

  PointT center_of_mass;
};

template <typename T>
class Octree {
  // Octree must be 3D, so this is fine
  // x,y,z, and a mass
  using PointT = Point<4, T>;

 public:
  Octree() = delete;

  explicit Octree(const PointT* input_data, const int n,
                  const OctreeParams<T> params)
      : root_(), params_(params), data_(input_data), data_size_(n) {}

  void BuildTree() {
    auto bodies = std::vector<IndexT>(data_size_);
    std::iota(std::begin(bodies), std::end(bodies), IndexT());

    root_ = BuildRecursive(params_.starting_box, bodies, 0);

    ComputeNodeMassRecursive(root_);

    if constexpr (true) {
      std::cout << "Tree Statistic: \n"
                << "\tNum leaf nodes: \t" << statistic_.num_leaf_nodes << '\n'
                << "\tNum branch nodes: \t" << statistic_.num_branch_nodes
                << '\n'
                << "\tMax Depth: \t" << statistic_.max_depth << '\n'
                << std::endl;
    }
  }

  OctreeStatistic GetStats() const { return statistic_; }
  OctreeParams<T> GetParams() const { return params_; }

  void LoadPayload(Point4F* leaf_node_content_table,
                   int* leaf_node_size_table) {
    LoadPayloadRecursive(root_, leaf_node_content_table, leaf_node_size_table);
  }

  Node<T>* BuildRecursive(const BoundingBox<T> box,
                          const std::vector<IndexT>& bodies, const int depth) {
    if (bodies.empty()) {
      // Should not happend?
      return nullptr;
    }

    const auto node = new Node<T>();

    node->bounding_box = box;

    if (bodies.size() <= params_.leaf_max_size)  // minimum is 1
    {
      ++statistic_.num_leaf_nodes;
      statistic_.max_depth = std::max(depth, statistic_.max_depth);

      // build as leaf node
      node->is_leaf = true;
      node->bodies = bodies;
      node->uid = GetNextLeafId();
    } else {
      ++statistic_.num_branch_nodes;

      // build as tree node. A.K.A, split the space
      node->is_leaf = false;
      node->uid = GetNextBranchId();

      const auto half_dimension = box.dimension / T(2.0);

      std::array<std::vector<IndexT>, 8> sub_bodies{};

      for (const auto idx : bodies) {
        const auto quadrant = DetermineQuadrant(
            box, data_[idx].data[X], data_[idx].data[Y], data_[idx].data[Z]);
        sub_bodies[quadrant].push_back(idx);
      }

      node->children[0] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] - half_dimension.data[X],
                          box.center.data[Y] - half_dimension.data[Y],
                          box.center.data[Z] - half_dimension.data[Z]}},
          sub_bodies[0], depth + 1);

      node->children[1] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] - half_dimension.data[X],
                          box.center.data[Y] - half_dimension.data[Y],
                          box.center.data[Z] + half_dimension.data[Z]}},
          sub_bodies[1], depth + 1);

      node->children[2] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] - half_dimension.data[X],
                          box.center.data[Y] + half_dimension.data[Y],
                          box.center.data[Z] - half_dimension.data[Z]}},
          sub_bodies[2], depth + 1);

      node->children[3] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] - half_dimension.data[X],
                          box.center.data[Y] + half_dimension.data[Y],
                          box.center.data[Z] + half_dimension.data[Z]}},
          sub_bodies[3], depth + 1);

      node->children[4] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] + half_dimension.data[X],
                          box.center.data[Y] - half_dimension.data[Y],
                          box.center.data[Z] - half_dimension.data[Z]}},
          sub_bodies[4], depth + 1);

      node->children[5] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] + half_dimension.data[X],
                          box.center.data[Y] - half_dimension.data[Y],
                          box.center.data[Z] + half_dimension.data[Z]}},
          sub_bodies[5], depth + 1);

      node->children[6] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] + half_dimension.data[X],
                          box.center.data[Y] + half_dimension.data[Y],
                          box.center.data[Z] - half_dimension.data[Z]}},
          sub_bodies[6], depth + 1);

      node->children[7] = BuildRecursive(
          BoundingBox<T>{half_dimension,
                         {box.center.data[X] + half_dimension.data[X],
                          box.center.data[Y] + half_dimension.data[Y],
                          box.center.data[Z] + half_dimension.data[Z]}},
          sub_bodies[7], depth + 1);
    }

    return node;
  }

  void ComputeNodeMassRecursive(Node<T>* cur) {
    if (cur == nullptr) {
      return;
    }

    if (cur->IsLeaf()) {
      if (!cur->bodies.empty()) {
        for (const auto idx : cur->bodies) {
          cur->node_mass += data_[idx].data[MASS];
          cur->node_weighted_pos.data[X] += data_[idx].data[X];
          cur->node_weighted_pos.data[Y] += data_[idx].data[Y];
          cur->node_weighted_pos.data[Z] += data_[idx].data[Z];
        }
      }
    } else {
      for (Node<T>* child : cur->children) {
        ComputeNodeMassRecursive(child);
      }

      for (Node<T>* child : cur->children) {
        if (child != nullptr) {
          cur->node_mass += child->node_mass;
          cur->node_weighted_pos += child->node_weighted_pos;
        }
      }
    }
  }

  void LoadPayloadRecursive(Node<T>* cur, Point4F* leaf_node_content_table,
                            int* leaf_node_size_table) {
    if (cur->IsLeaf()) {
      const auto offset = cur->uid * params_.leaf_max_size;

      for (auto i = 0u; i < cur->bodies.size(); ++i) {
        const auto idx = cur->bodies[i];

        leaf_node_content_table[offset + i] = data_[idx];
      }

      leaf_node_size_table[cur->uid] = static_cast<int>(cur->bodies.size());
    } else {
      // Compute and store center of mass to a member
      cur->center_of_mass = cur->node_weighted_pos / cur->node_mass;
      cur->center_of_mass.data[MASS] = cur->node_mass;

      for (Node<T>* child : cur->children) {
        if (child != nullptr) {
          LoadPayloadRecursive(child, leaf_node_content_table,
                               leaf_node_size_table);
        }
      }
    }
  }

  static int DetermineQuadrant(const BoundingBox<T> box, const T x, const T y,
                               const T z) {
    if (x < box.center.data[X]) {
      if (y < box.center.data[Y]) {
        if (z < box.center.data[Z]) {
          return 0;
        }
        return 1;
      }
      if (z < box.center.data[Z]) {
        return 2;
      }
      return 3;
    }
    if (y < box.center.data[Y]) {
      if (z < box.center.data[Z]) {
        return 4;
      }
      return 5;
    }
    if (z < box.center.data[Z]) {
      return 6;
    }
    return 7;
  }

  // Expose some APIs for Executor
 public:
  const Node<T>* GetRoot() const { return root_; }

 private:
  static int GetNextLeafId() {
    static int uid_counter = 0;
    return uid_counter++;
  }

  static int GetNextBranchId() {
    static int uid_counter = 0;
    return uid_counter++;
  }

  Node<T>* root_;

  OctreeParams<T> params_;
  OctreeStatistic statistic_;
  const PointT* data_;
  const int data_size_;
};
}  // namespace oct