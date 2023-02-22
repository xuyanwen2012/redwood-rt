#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "../knn/KDTree.hpp"
#include "../knn/Kernel.hpp"
#include "../knn/KnnSet.hpp"
#include "ReducerHandler.hpp"
#include "Redwood/Core.hpp"

inline std::shared_ptr<kdt::KdTree> tree_ref;

enum class ExecutionState { kWorking, kFinished };

struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

// Knn Algorithm
class Executor {
 public:
  Executor() = delete;

  // Thread id, i.e., [0, .., n_threads]
  // Stream id in the thread, i.e., [0, 1]
  // My id in the group executor, i.e., [0,...,1023]
  Executor(const int tid, const int stream_id, const int my_id)
      : state_(ExecutionState::kFinished),
        my_tid_(tid),
        my_stream_id_(stream_id),
        debug_uid_(my_id) {
    // values need experiment
    stack_.reserve(16);

    // When created,
    float* base_addr = rdc::GetResultAddr<float>(my_tid_, my_stream_id_);
    u_my_result_addr_ = base_addr + my_id;

    // std::cout << "\tCreated " << my_id << ": at addr " << u_my_result_addr_
    //           << std::endl;
  }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

  void StartQuery(const Point4F q) {
    my_query_point_ = q;
    stack_.clear();
    k_set_->Clear();
    cur_ = nullptr;
    Execute();
  }

  void Resume() { Execute(); }

 protected:
  void Execute() {
    if (state_ == ExecutionState::kWorking) goto my_resume_point;

    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          //
          // **** Reduction at Leaf Node (replaced with Redwood API) ****
          rdc::ReduceLeafNode(my_tid_, my_stream_id_, cur_->uid,
                              my_query_point_);
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
            KernelFunc(tree_ref->in_data_ref_[accessor_idx], my_query_point_);

        k_set_->Insert(dist);

        // **********************************

        const int axis = cur_->node_type.tree.axis;
        const float train = tree_ref->in_data_ref_[accessor_idx].data[axis];
        const kdt::Dir dir = my_query_point_.data[axis] < train
                                 ? kdt::Dir::kLeft
                                 : kdt::Dir::kRight;

        stack_.push_back({cur_, axis, train, dir});
        cur_ = cur_->GetChild(dir);
      }

      // We resume back from Break point, and now we are still in the branch
      // node, we can check if there's any thing left on the stack.
      if (!stack_.empty()) {
        const auto [last_cur, axis, train, dir] = stack_.back();
        stack_.pop_back();

        // Check if there is a possibility of the NN lies on the other half
        // If the difference between the query point and the other splitting
        // plane is greater than the current found minimum distance, then it
        // is impossible to have a NN there.

        Point4F a{};
        Point4F b{};
        a.data[axis] = my_query_point_.data[axis];
        b.data[axis] = train;
        const auto diff = KernelFunc(a, b);
        if (diff < k_set_->WorstDist()) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals, Write back to final results
    state_ = ExecutionState::kFinished;

    // k_set_->DebugPrint();
    // exit(0);
  }

 private:
  Point4F my_query_point_;

  // this should point to USM region, unique to each executor.
  // If k = 32,
  // Each address is 128 bytes (or 16 eight-byte words) apart.
 public:
  // Pointer to the USM address requested from the Backend.
  union {
    KnnSet<float, 1>* k_set_;
    float* u_my_result_addr_;
  };

  // Couroutine related
  std::vector<CallStackField> stack_;
  kdt::Node* cur_;
  ExecutionState state_;

  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;

 public:
  const int debug_uid_;
};

class CpuExecutor {
 public:
  CpuExecutor(const int tid) : my_tid_(tid) {}

  void StartQuery(const Point4F q) {
    my_query_point_ = q;
    k_set_.Clear();
    TraversalRecursive(tree_ref->GetRoot());
  }

  KnnSet<float, 1> k_set_;

 private:
  void TraversalRecursive(const kdt::Node* cur) {
    if (cur->IsLeaf()) {
      // k_set_->DebugPrint();

      const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
      for (int i = 0; i < tree_ref->GetParams().leaf_max_size; ++i) {
        const float dist = KernelFunc(leaf_addr[i], my_query_point_);
        k_set_.Insert(dist);
      }

    } else {
      // **** Reduction at tree node ****
      const unsigned accessor_idx =
          tree_ref->v_acc_[cur->node_type.tree.idx_mid];
      const float dist =
          KernelFunc(tree_ref->in_data_ref_[accessor_idx], my_query_point_);
      k_set_.Insert(dist);

      // **********************************

      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
      const auto dir = my_query_point_.data[axis] < train ? kdt::Dir::kLeft
                                                          : kdt::Dir::kRight;

      TraversalRecursive(cur->GetChild(dir));

      Point4F a{};
      Point4F b{};
      a.data[axis] = my_query_point_.data[axis];
      b.data[axis] = train;
      const auto diff = KernelFunc(a, b);
      if (diff < k_set_.WorstDist()) {
        TraversalRecursive(cur->GetChild(FlipDir(dir)));
      }
    }
  }

  Point4F my_query_point_;
  const int my_tid_;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "requires an input file (\"data/input_nn_1m_4f.dat\")\n";
    return EXIT_FAILURE;
  }

  const char* data_file = argv[1];
  const auto leaf_size = 96;
  const auto batch_size = 32;

  const auto [in, n] = mmap_file<Point4F>(data_file);

  // Inspect input data is correct
  for (int i = 0; i < 10; ++i) {
    std::cout << in[i] << std::endl;
  }

  std::cout << "Building Tree..." << std::endl;
  const kdt::KdtParams params{leaf_size};
  auto tree = std::make_shared<kdt::KdTree>(params, in, n);
  tree_ref = tree;

  std::cout << "Loading USM leaf node data..." << std::endl;
  rdc::InitReducers();

  // Now octree tree is comstructed, need to move leaf node data into USM
  const auto num_leaf_nodes = tree->GetStats().num_leaf_nodes;
  rdc::AllocateLeafNodeTable(num_leaf_nodes, leaf_size, false);
  tree->LoadPayload(rdc::LntDataAddr());

  // TODO: figure out a better way, not important right now
  // munmap_file(in, n);

  std::cout << "Making tasks..." << std::endl;

  static auto rand_point4f = []() {
    return Point4F{MyRand(0.0f, 1000.0f), MyRand(0.0f, 1000.0f),
                   MyRand(0.0f, 1000.0f), 1.0f};
  };
  const auto m = 1024;
  std::queue<Point4F> q_data;
  for (int i = 0; i < m; ++i) q_data.push(rand_point4f());

  std::cout << "Start Traversal " << std::endl;

  // Example:
  //   Created 0: at addr 0x100c60000
  //   Created 1: at addr 0x100c60080
  //   Created 2: at addr 0x100c60100
  //   Created 3: at addr 0x100c60180
  //   Created 4: at addr 0x100c60200
  //   Created 5: at addr 0x100c60280
  //   ...
  //
  constexpr int tid = 0;

  std::vector<float> final_results;
  final_results.reserve(m + 1);  // need to discard the first

  if constexpr (constexpr auto cpu = false; cpu) {
    // Just CPU traversal

    CpuExecutor exe{tid};

    const auto q = q_data.front();
    q_data.pop();

    exe.StartQuery(q);

    exe.k_set_.DebugPrint();

  } else {
    // Use redwood runtime
    std::vector<Executor> exe[rdc::kNumStreams];
    for (int stream_id = 0; stream_id < rdc::kNumStreams; ++stream_id) {
      exe[stream_id].reserve(batch_size);
      for (int i = 0; i < batch_size; ++i) {
        exe[stream_id].emplace_back(tid, stream_id, i);
      }
    }

    int cur_stream = 0;
    while (!q_data.empty()) {
      // Execute a group of 1024 executors at once, 1 step per executor
      // for (auto it = exe[cur_stream].begin(); it != exe[cur_stream].end();
      // ++it) {
      for (auto& exe : exe[cur_stream]) {
        if (exe.Finished()) {
          std::cout << exe.k_set_->WorstDist() << std::endl;

          // Make there is task in the queue
          if (q_data.empty()) {
            break;
          }

          const auto q = q_data.front();
          q_data.pop();

          exe.StartQuery(q);
        } else {
          exe.Resume();
        }
      }

      rdc::LuanchKernelAsync(tid, cur_stream);

      const auto next = rdc::NextStream(cur_stream);
      redwood::DeviceStreamSynchronize(next);

      // We could get results now
      // const auto result = rdc::GetResultValueUnchecked<float>(tid, next);
      // exe

      // Todo: this q_idx is not true, should be the last one
      // final_results.push_back(result);

      // Switch buffer ( A->B, B-A)
      cur_stream = next;
      rdc::ClearBuffer(tid, cur_stream);
    }
  }

  // TODO: Handle Remaining

  // rdc::LuanchKernelAsync(tid, cur_stream);
  // const auto next = rdc::NextStream(cur_stream);
  // redwood::DeviceStreamSynchronize(next);

  // for (int i = 0; i < m; ++i) {
  //   const auto q = final_results[i + 1];
  //   std::cout << i << ": " << q << std::endl;
  // }

  rdc::ReleaseReducers();
  return EXIT_SUCCESS;
}