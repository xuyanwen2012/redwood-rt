#pragma once

#include <array>
#include <numeric>
#include <vector>

#include "Redwood/Point.hpp"
#include "Redwood.hpp"

namespace oct {

#define X 0
#define Y 1
#define Z 2
#define MASS 3

using IndexT = int;

const int MAX_NODES = 50000;

template <typename T>
struct Range {
  int low;
  int high;
};

// For oct tree the bounding box is always 3D
template <typename T>
struct BoundingBox {
  Point<3, T> min;
  Point<3, T> max;

  static BoundingBox Unit() {
    return {
        Point<3, T>{0.0, 0.0, 0.0},  // min
        Point<3, T>{1000, 1000, 1000}   // max
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

  Node() : is_leaf(true), uid(-1) {
    for (int i = 0; i < 8; i++) {
      children[i] = -1;
    }
    return;
  }

  bool IsLeaf() const { return is_leaf; }

  std::array<int, 8> children;
  bool is_leaf;
  int uid;
};
/*
template <typename T>
class Array {
 private:
  T* elements;
  size_t size_;

 public:
  // Constructor
  Array(size_t size) : size_(size) { elements = new T[size]; }

  // Destructor
  ~Array() { delete[] elements; }

  // Accessor: Get element at index
  T& operator[](size_t index) { return elements[index]; }

  // Accessor: Get element at index (const version)
  const T& operator[](size_t index) const { return elements[index]; }

  // Accessor: Get size of the array
  size_t getSize() const { return size_; }
};
*/

BoundingBox<float> bounding_boxes[MAX_NODES];
float node_masses[MAX_NODES];
Point<4, float> center_of_mass[MAX_NODES];
Point<4, float> node_weighted_pos[MAX_NODES];
Node<Point4F> nodes[MAX_NODES];

  /*
Array<BoundingBox<float>> bounding_boxes = Array<BoundingBox<float>>(MAX_NODES);
Array<float> node_masses = Array<float>(MAX_NODES);
Array<Point<4, float>> center_of_mass = Array<Point<4, float>>(MAX_NODES);
Array<Point<4, float>> node_weighted_pos = Array<Point<4, float>>(MAX_NODES);
Array<Node<Point4F>> nodes = Array<Node<Point4F>>(MAX_NODES);
*/
std::array<std::pair<int, int>, MAX_NODES> range;

int next_node = 0;

int new_node(){
  int cur = next_node;
  next_node += 1;
  return cur;
}
template <typename T>
class Octree {
  // Octree must be 3D, so this is fine
  // x,y,z, and a mass
  using PointT = Point<4, T>;

 public:
  Octree() = delete;

  explicit Octree(PointT* input_data, const int n,
                  const OctreeParams<T> params)
      : root_(-1), params_(params), data_(input_data), data_size_(n) {}

  void BuildTree() {
    bodies_ = std::vector<IndexT>(data_size_);
    std::iota(std::begin(bodies_), std::end(bodies_), IndexT());

    std::cout << "build recursive" << std::endl;
    root_ = BuildRecursive(params_.starting_box, 0, data_size_ - 1, 0);
    std::cout << "compute node mass" << std::endl;
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

  int BuildRecursive(const BoundingBox<T> box, const int low, const int high,
                     const int depth) {
    int cur = new_node();
    bounding_boxes[cur] = box;
     //std::cout<<"cur: " <<cur <<std::endl;
    const size_t n = high - low;
    /*
      for (int i = 0; i < depth; i++) {
        std::cout << "-";
      }
      std::cout << "[" << cur << "]"
                << "low: " << low << "\thigh: " << high << "\tn: " << n
                << std::endl;
  */
    if (n <= params_.leaf_max_size) {
      ++statistic_.num_leaf_nodes;
      statistic_.max_depth = std::max(depth, statistic_.max_depth);
      nodes[cur].uid = GetNextLeafId();
      range[cur].first = low;
      range[cur].second = high;
    } else {
      ++statistic_.num_branch_nodes;
      nodes[cur].is_leaf = false;
      nodes[cur].uid = GetNextBranchId();
      BoundingBox<T> sub_boxes[8];
      split_box(box, sub_boxes);
      
      std::stable_sort(data_+ low, data_+low+n,
            [box](const PointT& a, const PointT& b) {
              const int group_id_a =
                  DetermineQuadrant(box, a.data[0], a.data[1], a.data[2]);
              const int group_id_b =
                  DetermineQuadrant(box, b.data[0], b.data[1], b.data[2]);
              if (group_id_a < group_id_b) return -1;
              if (group_id_a > group_id_b) return 1;
              return 0;
            });
            
        
      int count[8];
      for (int i = 0; i < 8; ++i) {
        count[i] = 0;
      }
      // std::array<std::vector<IndexT>, 8> sub_bodies{};

      for (auto i = low; i < high; ++i) {
        const auto quadrant = DetermineQuadrant(
            box, data_[i].data[X], data_[i].data[Y], data_[i].data[Z]);
        // sub_bodies[quadrant].push_back(bodies_[i]);
        count[quadrant] += 1;
      }
      /*
      int offset = low;
      for (const auto quadrant : sub_bodies) {
        for (auto idx : quadrant) {
          bodies_[offset] = idx;
          offset += 1;
        }
      }
      */

      int next_low = low;
      for(int i = 0; i < 8; ++i){
        const int next_high = next_low + count[i];
        nodes[cur].children[i] = BuildRecursive(sub_boxes[i], next_low, next_high, depth+1);
        next_low = next_high;
      }
      range[cur].first = range[nodes[cur].children[0]].first;
      range[cur].second = range[nodes[cur].children[7]].second;
    }
    return cur;
  }

  void ComputeNodeMassRecursive(int cur) {
    if (cur == -1) {
      return;
    }
    //std::cout<<"curr:" <<cur<<std::endl;
    if (nodes[cur].IsLeaf()) {
      if (range[cur].first != range[cur].second) {
        for (auto i = range[cur].first; i < range[cur].second; i++) {
          node_masses[cur] += data_[i].data[MASS];
          node_weighted_pos[cur].data[X] += data_[i].data[X];
          node_weighted_pos[cur].data[Y] += data_[i].data[Y];
          node_weighted_pos[cur].data[Z] += data_[i].data[Z];
        }
      }
    } else {
      for (int child : nodes[cur].children) {
        ComputeNodeMassRecursive(child);
      }
      for (int child : nodes[cur].children) {
        if (child != -1) {
          node_masses[cur] += node_masses[child];
          node_weighted_pos[cur] += node_weighted_pos[child];
        }
      }
    }
  }
  void LoadPayloadRecursive(int cur, Point4F* leaf_node_content_table,
                            int* leaf_node_size_table) {
    if (nodes[cur].IsLeaf()) {
      auto counter = 0;
      const auto offset = nodes[cur].uid * params_.leaf_max_size;

      for (auto i = range[cur].first; i < range[cur].second; ++i) {
        leaf_node_content_table[offset + counter] = data_[i];
        ++counter;
      }

      leaf_node_size_table[nodes[cur].uid] =
          static_cast<int>(range[cur].second - range[cur].first);
    } else {
      // Compute and store center of mass to a member
      center_of_mass[cur] = node_weighted_pos[cur] / node_masses[cur];
      center_of_mass[cur].data[MASS] = node_masses[cur];

      for (int child : nodes[cur].children) {
        if (child != -1) {
          LoadPayloadRecursive(child, leaf_node_content_table,
                               leaf_node_size_table);
        }
      }
    }
  }

  void split_box(const BoundingBox<T> box, BoundingBox<T> (&sub_boxes)[8]) {
    const float x_min = box.min.data[0];
    const float y_min = box.min.data[1];
    const float z_min = box.min.data[2];
    const float x_max = box.max.data[0];
    const float y_max = box.max.data[1];
    const float z_max = box.max.data[2];
    const float x_mid = (x_min + x_max) / 2.0f;
    const float y_mid = (y_min + y_max) / 2.0f;
    const float z_mid = (z_min + z_max) / 2.0f;
    sub_boxes[0] =
        BoundingBox<float>{{x_min, y_min, z_min}, {x_mid, y_mid, z_mid}};
    sub_boxes[1] =
        BoundingBox<float>{{x_mid, y_min, z_min}, {x_max, y_mid, z_mid}};
    sub_boxes[2] =
        BoundingBox<float>{{x_min, y_mid, z_min}, {x_mid, y_max, z_mid}};
    sub_boxes[3] =
        BoundingBox<float>{{x_mid, y_mid, z_min}, {x_max, y_max, z_mid}};
    sub_boxes[4] =
        BoundingBox<float>{{x_min, y_min, z_mid}, {x_mid, y_mid, z_max}};
    sub_boxes[5] =
        BoundingBox<float>{{x_mid, y_min, z_mid}, {x_max, y_mid, z_max}};
    sub_boxes[6] =
        BoundingBox<float>{{x_min, y_mid, z_mid}, {x_mid, y_max, z_max}};
    sub_boxes[7] =
        BoundingBox<float>{{x_mid, y_mid, z_mid}, {x_max, y_max, z_max}};
  }
  static int DetermineQuadrant( const BoundingBox<T>& box, const T x, const T y,
                               const T z) {
    const float x_mid = (box.min.data[0] + box.max.data[0]) / 2.0f;
    const float y_mid = (box.min.data[1] + box.max.data[1]) / 2.0f;
    const float z_mid = (box.min.data[2] + box.max.data[2]) / 2.0f;
    int quadrant;
    if (x < x_mid) {
      if (y < y_mid) {
        if (z < z_mid) {
          quadrant = 0;
        } else {
          quadrant = 4;
        }
      } else {
        if (z < z_mid) {
          quadrant = 2;
        } else {
          quadrant = 6;
        }
      }
    } else {
      if (y < y_mid) {
        if (z < z_mid) {
          quadrant = 1;
        } else {
          quadrant = 5;
        }
      } else {
        if (z < z_mid) {
          quadrant = 3;
        } else {
          quadrant = 7;
        }
      }
    }
    return quadrant;
  }

  // Expose some APIs for Executor
 public:
  const int GetRoot() const { return root_; }

 private:
  static int GetNextLeafId() {
    static int uid_counter = 0;
    return uid_counter++;
  }

  static int GetNextBranchId() {
    static int uid_counter = 0;
    return uid_counter++;
  }

  int root_;
  OctreeParams<T> params_;
  OctreeStatistic statistic_;
  std::vector<IndexT> bodies_;
  PointT* data_;
  const int data_size_;
};
}  // namespace oct