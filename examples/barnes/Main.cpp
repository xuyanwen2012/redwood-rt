#include <algorithm>
#include <array>
#include <numeric>
#include <queue>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "DuetReducer.hpp"
#include "Kernel.hpp"
#include "Octree.hpp"
#include "Reducer.hpp"
#include "Redwood.hpp"

// Redwood user need to specify which reducer to use (barnes/knn) and its types

using MyReducer = rdc::DoubleBufferReducer<Point4F, float>;
// using MyReducer = rdc::DuetBarnesReducer<Point4F, float>;

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

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

    // Notify Reducer to
    // In case of FPGA, it will register the anchor point (q) into the registers
    MyReducer::SetQuery(my_tid_, my_stream_id_, q);
  }

  // Main Barnes-Hut Traversal Algorithm, annotated with Redwood APIs
  void TraverseRecursive(const oct::Node<float>* cur) {
    if (cur->IsLeaf()) {
      if (cur->bodies.empty()) return;

      // ------------------------------------------------------------
      MyReducer::ReduceLeafNode(my_tid_, my_stream_id_, cur->uid);
      // ------------------------------------------------------------

      ++stats_.leaf_node_reduced;
    } else if (const auto my_theta = ComputeThetaValue(cur, my_q_);
               my_theta < theta_) {
      ++stats_.branch_node_reduced;

      // ------ TODO: lets' not reduce this on accelerator ----------
      constexpr auto my_functor = MyFunctor();
      const auto dist = my_functor(cur->CenterOfMass(), my_q_);
      // MyReducer::ReduceBranchNode(my_tid_, my_stream_id_,
      // cur->CenterOfMass());
      // ------------------------------------------------------------

    } else
      for (const auto child : cur->children)
        if (child != nullptr) TraverseRecursive(child);
  }

  ExecutorStats GetStats() const { return stats_; }

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

 private:
  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;
  static float theta_;  // = 0.2f;
  ExecutorStats stats_;

  Point4F my_q_;

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

  std::cout << "Loading USM leaf node data..." << std::endl;

  // Initialize Backend (find device etc., warmup), Reducer(double buffer), and
  // Executor(tree algorithm)
  redwood::Init();
  MyReducer::InitReducers();
  Executor::SetThetaValue(theta);

  // Now octree tree is comstructed, need to move leaf node data into USM
  const auto num_leaf_nodes = tree.GetStats().num_leaf_nodes;

  rdc::AllocateLeafNodeTable(num_leaf_nodes, leaf_size, true);
  tree.LoadPayload(rdc::LntDataAddr(), rdc::LntSizesAddr());

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
  final_results.reserve(m + 1);  // need to discard the first

  // Assume in future version this tid will be generated?
  std::cout << "Start Traversal " << std::endl;
  constexpr int tid = 0;

  // ------------------- CUDA ------------------------------------
  TimeTask("Traversal", [&] {
    int cur_stream = 0;
    Executor exe[redwood::kNumStreams]{{tid, 0}, {tid, 1}};

    while (!q_data.empty()) {
      const auto q = q_data.front();

      // Set anchor
      exe[cur_stream].NewTask(q);

      // Traverse tree and collect data
      exe[cur_stream].TraverseRecursive(tree.GetRoot());

      std::cout << "\tl: " << exe[cur_stream].GetStats().leaf_node_reduced
                << "\tb: " << exe[cur_stream].GetStats().branch_node_reduced
                << std::endl;

      MyReducer::LuanchKernelAsync(tid, cur_stream);

      // Synchronize the next stream
      const auto next = MyReducer::NextStream(cur_stream);
      redwood::DeviceStreamSynchronize(next);

      // Read results that were computed and synchronized before
      const auto result = *MyReducer::GetResultAddr(tid, next);

      // Todo: this q_idx is not true, should be the last one
      final_results.push_back(result);

      // Switch buffer ( A->B, B-A)
      cur_stream = next;
      MyReducer::ClearBuffer(tid, cur_stream);

      q_data.pop();
    }

    // When q are finished, remember to Synchronize the last bit;
    redwood::DeviceSynchronize();
    const auto next = MyReducer::NextStream(cur_stream);

    const auto result = *MyReducer::GetResultAddr(tid, next);
    final_results.push_back(result);
  });

  // -------------------------------------------------------------

  for (int i = 0; i < m; ++i) {
    const auto q = final_results[i + 1];
    std::cout << i << ": " << q << std::endl;
  }

  rdc::FreeLeafNodeTalbe();
  MyReducer::ReleaseReducers();

  return EXIT_SUCCESS;
}