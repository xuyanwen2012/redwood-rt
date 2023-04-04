#pragma once

#include "../nn/GlobalVars.hpp"
#include "../nn/KDTree.hpp"
#include "ReducerHandler.hpp"

using Task = std::pair<int, Point4F>;

enum class ExecutionState { kWorking, kFinished };

struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

// Nn/Knn Algorithm
template <typename Functor>
class Executor {
 public:
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
    my_assigned_result_addr = rdc::RequestResultAddr(tid, stream_id, uid);
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

  _NODISCARD float CpuTraverse() {
    result_set->Reset();
    TraversalRecursive(tree_ref->root_);
    return result_set->WorstDist();
  }

 protected:
  void Execute() {
    constexpr Functor functor;

    if (state_ == ExecutionState::kWorking) goto my_resume_point;
    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          // **** Reduction at Leaf Node (replaced with Redwood API) ****

          rdc::ReduceLeafNode(my_tid_, my_stream_id_, my_task_, cur_->uid);

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

        // Recursion 1
        stack_.push_back({cur_, axis, train, dir});
        cur_ = cur_->GetChild(dir);
      }

      if (!stack_.empty()) {
        const auto [last_cur, axis, train, dir] = stack_.back();
        stack_.pop_back();

        if (const auto diff = functor(my_task_.second.data[axis], train);
            diff < result_set->WorstDist()) {
          // Recursion 2
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals
    state_ = ExecutionState::kFinished;

    final_results1[my_task_.first] = result_set->WorstDist();
  }

  void TraversalRecursive(const kdt::Node* cur) {
    constexpr Functor functor;

    if (cur->IsLeaf()) {
      // **** Reduction at leaf node ****
      const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
      for (int i = 0; i < rdc::stored_max_leaf_size; ++i) {
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

  union {
    float* my_assigned_result_addr;
    KnnSet<float, 1>* result_set = nullptr;
  };

  // Couroutine related
  std::vector<CallStackField> stack_;
  kdt::Node* cur_;
  ExecutionState state_;

  // Store some reference used (const)
  int my_tid_;
  int my_stream_id_;
  int my_uid_;
};
