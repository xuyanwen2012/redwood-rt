#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "../Utils.hpp"
#include "../cxxopts.hpp"
#include "AppParams.hpp"
#include "DistanceMetrics.hpp"
#include "KDTree.hpp"
#include "LoadFile.hpp"
#include "ReducerHandler.hpp"

// Global vars
std::shared_ptr<kdt::KdTree> tree_ref;

// Debug
std::vector<std::vector<int>> leaf_node_visited1;
std::vector<std::vector<int>> leaf_node_visited2;
std::vector<float> final_results1;
std::vector<float> final_results2;

// std::vector<Point4F> lnt;
// _NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
//   return lnt.data() + node_idx * app_params.max_leaf_size;
// }

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

// // For Nearest Neighbor
// struct KnnSet {
//   void Insert(const float value) {
//     if (value < dat) dat = value;
//   }

//   void Reset() { dat = std::numeric_limits<float>::max(); }

//   _NODISCARD float WorstDist() const { return dat; }

//   float dat;
// };

// Knn Algorithm
class Executor {
 public:
  Executor() = delete;

  // Thread id, i.e., [0, .., n_threads]
  // Stream id in the thread, i.e., [0, 1]
  // My id in the group executor, i.e., [0,...,1023]
  Executor(const int tid, const int stream_id, const int uid)
      : cur_(),
        state_(ExecutionState::kFinished),
        my_tid_(tid),
        my_stream_id_(stream_id),
        my_uid_(uid) {
    stack_.reserve(16);
    my_assigned_result_addr = rdc::RequestResultAddr(stream_id, uid);

    // std::cout << my_uid_ << ": " << my_assigned_result_addr << std::endl;
  }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

  void SetQuery(const Task& task) { my_task_ = task; }

  void StartQuery() {
    stack_.clear();
    result_set->Reset();
    Execute();
  }

  void Resume() { Execute(); }

  void CpuTraverse() {
    result_set->Reset();

    TraversalRecursive(tree_ref->root_);

    final_results1[my_task_.first] = result_set->WorstDist();
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

          // Redwood ReduceLeaf
          rdc::ReduceLeafNode(my_stream_id_, my_task_, cur_->uid);

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

        result_set->Insert(dist);
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

        if (const auto diff = functor(my_task_.second.data[axis], train);
            diff < result_set->WorstDist()) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals
    state_ = ExecutionState::kFinished;

    final_results2[my_task_.first] = result_set->WorstDist();
  }

  void TraversalRecursive(const kdt::Node* cur) {
    constexpr dist::Euclidean functor;

    if (cur->IsLeaf()) {
      leaf_node_visited1[my_task_.first].push_back(cur->uid);

      // **** Reduction at leaf node ****
      const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
      for (int i = 0; i < app_params.max_leaf_size; ++i) {
        const float dist = functor(leaf_addr[i], my_task_.second);
        result_set->Insert(dist);
      }
      // **********************************
    } else {
      // **** Reduction at tree node ****
      const unsigned accessor_idx =
          tree_ref->v_acc_[cur->node_type.tree.idx_mid];
      const float dist =
          functor(tree_ref->in_data_ref_[accessor_idx], my_task_.second);
      result_set->Insert(dist);
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
          diff < result_set->WorstDist()) {
        TraversalRecursive(cur->GetChild(FlipDir(dir)));
      }
    }
  }

 public:
  // Current processing task and its result (kSet)
  Task my_task_;

  // KnnSet k_set_;

  union {
    float* my_assigned_result_addr;
    KnnSet* result_set = nullptr;
  };

  // Couroutine related
  std::vector<CallStackField> stack_;
  kdt::Node* cur_;
  ExecutionState state_;

  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;
  const int my_uid_;
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

  std::cout << app_params << std::endl;

  const auto in_data = load_data_from_file<Point4F>(data_file);
  const auto n = in_data.size();

  // Debug
  leaf_node_visited1.resize(app_params.m);
  leaf_node_visited2.resize(app_params.m);
  final_results1.resize(app_params.m);
  final_results2.resize(app_params.m);

  // Input (inspection)
  for (int i = 0; i < 10; i++) std::cout << in_data[i] << "\n";
  std::cout << std::endl;

  // Query (x2)
  std::queue<Task> q1_data;
  for (int i = 0; i < app_params.m; ++i) q1_data.emplace(i, RandPoint());
  std::queue q2_data(q1_data);

  rdc::Init(app_params.batch_size);

  // Build tree
  {
    const kdt::KdtParams params{app_params.max_leaf_size};
    tree_ref = std::make_shared<kdt::KdTree>(params, in_data.data(), n);

    const auto num_leaf_nodes = tree_ref->GetStats().num_leaf_nodes;

    rdc::lnt.resize(num_leaf_nodes * app_params.max_leaf_size);
    tree_ref->LoadPayload(rdc::lnt.data());
  }

  // Pure CPU traverse
  {
    Executor cpu_exe{0, 0, 0};

    while (!q1_data.empty()) {
      cpu_exe.SetQuery(q1_data.front());
      cpu_exe.CpuTraverse();
      q1_data.pop();
    }

    // PrintLeafNodeVisited(leaf_node_visited1);
    // PrintFinalResult(final_results1);
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
        exes[stream_id].emplace_back(tid, stream_id, i);
      }
    }

    auto cur_stream = 0;
    while (!q2_data.empty()) {
      for (auto it = exes[cur_stream].begin(); it != exes[cur_stream].end();) {
        if (it->Finished()) {
          if (!q2_data.empty()) {
            const auto q = q2_data.front();
            q2_data.pop();

            it->SetQuery(q);
            it->StartQuery();
          }

          ++it;
        } else {
          it->Resume();
          if (it->Finished()) {
            // Do not increment , let the same executor (it) take another task
          } else {
            ++it;
          }
        }
      }

      rdc::LaunchAsyncWorkQueue(cur_stream);

      // switch to next
      cur_stream = (cur_stream + 1) % num_streams;

      // redwood::DeviceStreamSynchronize(cur_stream);

      rdc::ResetBuffer(tid, cur_stream);
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

      rdc::LaunchAsyncWorkQueue(cur_stream);

      const auto next = (cur_stream + 1) % num_streams;
      cur_stream = next;
      rdc::buffers[cur_stream].Reset();

      // Both stream must complete
      need_work = false;
      for (const int i : num_incomplete) need_work |= i > 0;
    } while (need_work);

    // PrintLeafNodeVisited(leaf_node_visited2);
    // PrintFinalResult(final_results2);
  }

  // for (std::size_t i = 0; i < m; ++i)
  //{
  //	const auto& inner1 = leaf_node_visited1[i];
  //	const auto& inner2 = leaf_node_visited2[i];

  //	std::cout << "Mismatched values in vector " << i << ":\n";

  //	for (std::size_t j = 0; j < inner1.size(); ++j)
  //	{
  //		if (j >= inner2.size() || inner1[j] != inner2[j])
  //		{
  //			std::cout << inner1[j] << '\n';
  //		}
  //	}

  //	for (std::size_t j = inner1.size(); j < inner2.size(); ++j)
  //	{
  //		std::cout << inner2[j] << '\n';
  //	}

  //	std::cout << '\n';
  //}

  // Find the first mismatch between the two vectors

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

  rdc::Release();
  return 0;
}
