#include <algorithm>
#include <array>
#include <numeric>
#include <queue>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "../barnes/Kernel.hpp"
#include "../barnes/Octree.hpp"
#include "Redwood/Point.hpp"

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

std::vector<Point4F> lnt_data;
std::vector<int> lnt_sizes;

// BH Algorithm
class Executor {
 public:
  Executor(const int tid, const int stream_id)
      : my_tid_(tid), my_stream_id_(stream_id) {}

  static void SetThetaValue(const float theta) { theta_ = theta; }

  void NewTask(const Point4F q) {
    // Clear executor's data
    stats_.leaf_node_reduced = 0;
    stats_.branch_node_reduced = 0;
    cached_result_ = 0.0f;

    // This is a host copy of q point, so some times the traversal doesn't have
    // to bother with accessing USM
    my_q_ = q;
  }

  void TraverseRecursiveCpu(const oct::Node<float>* cur) {
    if (cur->IsLeaf()) {
      if (cur->bodies.empty()) return;
      cached_result_ += ReduceLeafsCpu(my_q_, cur->uid);
      ++stats_.leaf_node_reduced;
    } else {
      if (const auto my_theta = ComputeThetaValue(cur, my_q_);
          my_theta < theta_) {
        ++stats_.branch_node_reduced;
        // cached_result_ += KernelFunc(cur->CenterOfMass(), my_q_);
      } else
        for (const auto child : cur->children)
          if (child != nullptr) TraverseRecursiveCpu(child);
    }
  }

  float GetResult() const { return cached_result_; }

 protected:
  static float ComputeThetaValue(const oct::Node<float>* node,
                                 const Point4F pos) {
    const auto com = node->CenterOfMass();
    auto norm_sqr = 1e-9f;

    // Use only the first three property (x, y, z) for this theta compuation
    for (int i = 0; i < 3; ++i) {
      const auto diff = com.data[i] - pos.data[i];
      norm_sqr += diff * diff;
    }

    const auto norm = sqrtf(norm_sqr);
    return node->bounding_box.dimension.data[0] / norm;
  }

  static float ReduceLeafsCpu(const Point4F& q, int leaf_idx) {
    constexpr auto leaf_size = 64;
    const auto addr = &lnt_data[leaf_idx * leaf_size];

    float sum{};
    for (int i = 0; i < leaf_size; ++i) {
      sum += KernelFunc(addr[i], q);
    }

    return sum;
  }

 private:
  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;
  static float theta_;  // = 0.2f;

  Point4F my_q_;

  ExecutorStats stats_;

 public:
  // Used on the CPU side
  float cached_result_;
};

float Executor::theta_ = 0.2f;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "requires an input file (\"data/input_bh_2m_4f.dat\")\n";
    return EXIT_FAILURE;
  }

  const char* data_file = argv[1];
  const auto leaf_size = 64;
  const auto theta = 0.2f;

  const auto [in, n] = mmap_file<Point4F>(data_file);

  // Inspect input data is correct
  for (int i = 0; i < 10; ++i) {
    std::cout << in[i] << std::endl;
  }

  std::cout << "Building Tree..." << std::endl;

  const oct::BoundingBox<float> universe{
      Point3F{1000.0f, 1000.0f, 1000.0f},
      Point3F{500.0f, 500.0f, 500.0f},
  };

  const oct::OctreeParams<float> params{theta, leaf_size, universe};
  oct::Octree<float> tree(in, static_cast<int>(n), params);

  tree.BuildTree();

  // Now octree tree is comstructed, need to move leaf node data into USM
  const auto num_leaf_nodes = tree.GetStats().num_leaf_nodes;
  lnt_data.resize(num_leaf_nodes * leaf_size);
  lnt_sizes.resize(num_leaf_nodes);
  tree.LoadPayload(lnt_data.data(), lnt_sizes.data());

  munmap_file(in, n);

  std::cout << "Making tasks" << std::endl;

  static auto rand_point4f = []() {
    return Point4F{MyRand(0.0f, 1000.0f), MyRand(0.0f, 1000.0f),
                   MyRand(0.0f, 1000.0f), 1.0f};
  };

  const auto m = 32;
  std::queue<Point4F> q_data;
  for (int i = 0; i < m; ++i) q_data.push(rand_point4f());

  std::vector<float> final_results;
  final_results.reserve(m);  // need to discard the first

  Executor::SetThetaValue(theta);
  TimeTask("Cpu Traversal", [&] {
    Executor cpu_exe(0, 0);
    while (!q_data.empty()) {
      const auto q = q_data.front();

      cpu_exe.NewTask(q);
      cpu_exe.TraverseRecursiveCpu(tree.GetRoot());
      final_results.push_back(cpu_exe.cached_result_);

      q_data.pop();
    }
  });

  for (int i = 0; i < m; ++i) {
    const auto q = final_results[i];
    std::cout << i << ": " << q << std::endl;
  }

  return EXIT_SUCCESS;
}