#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "../cxxopts.hpp"
#include "AppParams.hpp"
#include "DistanceMetrics.hpp"
#include "LoadFile.hpp"
#include "Redwood/Macros.hpp"
#include "Redwood/Point.hpp"

namespace kdt {
class KdTree;
}

// Global vars
std::shared_ptr<kdt::KdTree> tree_ref;

// Debug
std::vector<std::vector<int>> leaf_node_visited1;
std::vector<std::vector<int>> leaf_node_visited2;
std::vector<float> final_results1;
std::vector<float> final_results2;

std::vector<Point4F> lnt;
_NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
  return lnt.data() + node_idx * app_params.max_leaf_size;
}

namespace kdt {
enum class Dir { kLeft = 0, kRight };

inline Dir FlipDir(const Dir dir) {
  return dir == Dir::kLeft ? Dir::kRight : Dir::kLeft;
}

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
    } tree;
  } node_type;

  Node* left_child;
  Node* right_child;
  int uid;  // In this version this is used only for leaf nodes.
};

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

class KdTree {
  using T = Point4F;

 public:
  KdTree() = delete;

  explicit KdTree(const KdtParams params, const T* in_data, const int n)
      : root_(), in_data_ref_(in_data), params_(params) {
    BuildTree(n);
  }

  void BuildTree(const int size) {
    v_acc_.resize(size);
    std::iota(v_acc_.begin(), v_acc_.end(), 0u);

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
  _NODISCARD const Node* GetRoot() const { return root_; }

  Node* BuildRecursive(const int left_idx, const int right_idx,
                       const int depth) {
    const auto node = new Node;

    if (right_idx - left_idx <= params_.leaf_max_size)  // minimum is 1
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

  void LoadPayloadRecursive(const Node* cur, T* usm_leaf_node_table) {
    if (cur->IsLeaf()) {
      auto counter = 0;
      const auto offset = cur->uid * params_.leaf_max_size;

      for (auto i = cur->node_type.leaf.idx_left;
           i <= cur->node_type.leaf.idx_right; ++i) {
        const auto idx = v_acc_[i];
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
  Node* root_;
  std::vector<int> v_acc_;

  // Datasets (ref to Input Data, and the Node Contents)
  // Note: dangerous, do not use after LoadPayload
  const T* in_data_ref_;

  // Statistics informations for/of the tree construction
  KdtParams params_;
  KdtStatistic statistic_;
};
}  // namespace kdt

static float MyRand(const float min = 0.0f, const float max = 1.0f) {
  // 114514 and 233
  static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution distribution(min, max);
  return distribution(generator);
}

Point4F RandPoint() {
  return {
      MyRand(0, 1024),
      MyRand(0, 1024),
      MyRand(0, 1024),
      MyRand(0, 1024),
  };
}

using Task = std::pair<int, Point4F>;

enum class ExecutionState { kWorking, kFinished };

struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

// For Nearest Neighbor
struct KnnSet {
  void Insert(const float value) {
    if (value < dat) dat = value;
  }

  void Reset() { dat = std::numeric_limits<float>::max(); }

  _NODISCARD float WorstDist() const { return dat; }

  float dat;
};

// Knn Algorithm
class Executor {
 public:
  // Executor() = delete;

  // Thread id, i.e., [0, .., n_threads]
  // Stream id in the thread, i.e., [0, 1]
  // My id in the group executor, i.e., [0,...,1023]
  Executor(const int tid, const int stream_id)
      : k_set_(),
        cur_(),
        state_(ExecutionState::kFinished),
        my_tid_(tid),
        my_stream_id_(stream_id) {
    stack_.reserve(16);
  }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

  void SetQuery(const Task& task) { my_task_ = task; }

  void StartQuery() {
    stack_.clear();
    k_set_.Reset();
    cur_ = nullptr;
    Execute();
  }

  void Resume() { Execute(); }

  void CpuTraverse() {
    k_set_.Reset();
    TraversalRecursive(tree_ref->root_);
    final_results1[my_task_.first] = k_set_.WorstDist();
  }

 protected:
  void Execute() {
    constexpr dist::Euclidean functor;

    if (state_ == ExecutionState::kWorking) goto my_resume_point;
    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          // **** Reduction at Leaf Node (replaced with Redwood API) ****
          leaf_node_visited2[my_task_.first].push_back(cur_->uid);

          for (int i = 0; i < app_params.max_leaf_size; ++i) {
            const float dist =
                functor(LntDataAddrAt(cur_->uid)[i], my_task_.second);
            k_set_.Insert(dist);
          }
          // ****************************

          // **** Coroutine Reuturn (API) ****
          return;
        my_resume_point:
          // ****************************

          cur_ = nullptr;
          continue;
        }

        // **** Reduction at tree node ****
        const unsigned accessor_idx =
            tree_ref->v_acc_[cur_->node_type.tree.idx_mid];
        const float dist =
            functor(tree_ref->in_data_ref_[accessor_idx], my_task_.second);
        k_set_.Insert(dist);
        // **********************************

        // Determine which child node to traverse next
        const auto axis = cur_->node_type.tree.axis;
        const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
        const auto dir = my_task_.second.data[axis] < train ? kdt::Dir::kLeft
                                                            : kdt::Dir::kRight;

        stack_.push_back({cur_, axis, train, dir});
        cur_ = cur_->GetChild(dir);
      }

      if (!stack_.empty()) {
        const auto [last_cur, axis, train, dir] = stack_.back();
        stack_.pop_back();

        const auto diff = functor(my_task_.second.data[axis], train);
        if (diff < k_set_.WorstDist()) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals
    state_ = ExecutionState::kFinished;

    final_results2[my_task_.first] = k_set_.WorstDist();
  }

  void TraversalRecursive(const kdt::Node* cur) {
    constexpr dist::Euclidean functor;

    if (cur->IsLeaf()) {
      leaf_node_visited1[my_task_.first].push_back(cur->uid);

      // **** Reduction at leaf node ****
      const auto leaf_addr = LntDataAddrAt(cur->uid);
      for (int i = 0; i < app_params.max_leaf_size; ++i) {
        const float dist = functor(leaf_addr[i], my_task_.second);
        k_set_.Insert(dist);
      }
      // **********************************
    } else {
      // **** Reduction at tree node ****
      const unsigned accessor_idx =
          tree_ref->v_acc_[cur->node_type.tree.idx_mid];
      const float dist =
          functor(tree_ref->in_data_ref_[accessor_idx], my_task_.second);
      k_set_.Insert(dist);
      // **********************************

      // Determine which child node to traverse next
      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
      const auto dir = my_task_.second.data[axis] < train ? kdt::Dir::kLeft
                                                          : kdt::Dir::kRight;

      // Will update 'k_dist' (dependency)
      TraversalRecursive(cur->GetChild(dir));

      // Check if we need to traverse the other side (optional)
      if (const auto diff = functor(my_task_.second.data[axis], train);
          diff < k_set_.WorstDist()) {
        TraversalRecursive(cur->GetChild(FlipDir(dir)));
      }
    }
  }

 public:
  // Current processing task and its result (kSet)
  Task my_task_;
  KnnSet k_set_;

  // Couroutine related
  std::vector<CallStackField> stack_;
  kdt::Node* cur_;
  ExecutionState state_;

  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;
};

void PrintLeafNodeVisited(const std::vector<std::vector<int>>& d) {
  for (auto i = 0u; i < d.size(); ++i) {
    std::cout << "Query " << i << ": [";
    for (const auto& elem : d[i]) {
      std::cout << elem;
      if (elem != d[i].back()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

void PrintFinalResult(const std::vector<float>& d) {
  for (auto i = 0u; i < d.size(); ++i)
    std::cout << "Query " << i << ": " << d[i] << '\n';
  std::cout << std::endl;
}

int main(int argc, char** argv) {
  cxxopts::Options options("Nearest Neighbor (NN)",
                           "Redwood NN demo implementation");
  options.add_options()("f,file", "File name", cxxopts::value<std::string>())(
      "q,query", "Num to Query", cxxopts::value<int>()->default_value("16384"))(
      "p,thread", "Num Thread", cxxopts::value<int>()->default_value("1"))(
      "l,leaf", "Leaf node size", cxxopts::value<int>()->default_value("32"))(
      "b,batch_size", "Batch Size",
      cxxopts::value<int>()->default_value("1024"))(
      "c,cpu", "Enable Cpu Baseline",
      cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

  options.parse_positional({"file", "query"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (!result.count("file")) {
    std::cerr << "requires an input file (\"data/input_nn_1m_4f.dat\")\n";
    std::cout << options.help() << std::endl;
    exit(EXIT_FAILURE);
  }

  // Config
  const auto data_file = result["file"].as<std::string>();
  app_params.max_leaf_size = result["leaf"].as<int>();
  app_params.m = result["query"].as<int>();
  app_params.batch_size = result["batch_size"].as<int>();
  app_params.num_threads = result["thread"].as<int>();
  app_params.cpu = result["cpu"].as<bool>();

  // Loaded
  //   const auto m = 128;

  const auto in_data = load_data_from_file<Point4F>(data_file);
  const auto n = in_data.size();

  // Debug
  leaf_node_visited1.resize(app_params.m);
  leaf_node_visited2.resize(app_params.m);
  final_results1.resize(app_params.m);
  final_results2.resize(app_params.m);

  // Input
  for (int i = 0; i < 10; i++) std::cout << in_data[i] << "\n";
  std::cout << std::endl;

  // Query (x2)
  std::queue<Task> q1_data;
  for (int i = 0; i < app_params.m; ++i) q1_data.emplace(i, RandPoint());
  std::queue q2_data(q1_data);

  // Build tree
  {
    const kdt::KdtParams params{app_params.max_leaf_size};
    tree_ref = std::make_shared<kdt::KdTree>(params, in_data.data(), n);

    const auto num_leaf_nodes = tree_ref->GetStats().num_leaf_nodes;

    lnt.resize(num_leaf_nodes * app_params.max_leaf_size);
    tree_ref->LoadPayload(lnt.data());
  }

  // Pure CPU traverse
  {
    Executor cpu_exe{0, 0};

    while (!q1_data.empty()) {
      cpu_exe.SetQuery(q1_data.front());
      cpu_exe.CpuTraverse();
      q1_data.pop();
    }

    PrintLeafNodeVisited(leaf_node_visited1);
    PrintFinalResult(final_results1);
  }

  std::cout << std::endl;

  // Traverser traverse (double buffer)
  {
    constexpr auto tid = 0;

    constexpr auto num_streams = 2;

    std::vector<Executor> exes[num_streams];
    for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
      exes[stream_id].reserve(app_params.batch_size);
      for (int i = 0; i < app_params.batch_size; ++i) {
        exes[stream_id].emplace_back(tid, stream_id);
      }
    }

    auto cur_stream = 0;
    while (!q2_data.empty()) {
      for (auto& exe : exes[cur_stream]) {
        if (exe.Finished()) {
          if (!q2_data.empty()) {
            const auto q = q2_data.front();
            q2_data.pop();

            exe.SetQuery(q);
            exe.StartQuery();
          }
        } else {
          exe.Resume();
        }
      }

      const auto next = (cur_stream + 1) % num_streams;
      // redwood::DeviceStreamSynchronize(next);
      cur_stream = next;
      // rdc::ClearBuffer(tid, cur_stream);
    }

    // Still some remaining
    int num_incomplete[num_streams];
    bool need_work;
    do {
      num_incomplete[cur_stream] = 0;
      for (auto& ex : exes[cur_stream]) {
        if (!ex.Finished()) {
          ex.Resume();
          ++num_incomplete[cur_stream];
        }
      }

      const auto next = (cur_stream + 1) % num_streams;
      cur_stream = next;

      need_work = false;
      for (const int i : num_incomplete) need_work |= i > 0;
    } while (need_work);

    PrintLeafNodeVisited(leaf_node_visited2);
    PrintFinalResult(final_results2);
  }

  // Print the indices and values of the mismatched elements
  if (const auto [fst, snd] = std::mismatch(
          final_results1.begin(), final_results1.end(), final_results2.begin());
      fst != final_results1.end()) {
    const auto index = std::distance(final_results1.begin(), fst);
    std::cout << "Mismatch at index " << index << ": " << *fst << " vs. "
              << *snd << "\n";
  } else {
    std::cout << "Vectors are equal.\n";
  }

  return 0;
}
