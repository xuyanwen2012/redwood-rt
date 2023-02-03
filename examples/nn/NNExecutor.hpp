#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "../KDTree.hpp"
#include "ExecutorManager.hpp"

struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

struct DummyExecutor {
  DummyExecutor() = delete;
  DummyExecutor(const int id) : state(ExecutionState::kFinished), tid(id) {}

  bool Finished() const { return state == ExecutionState::kFinished; }

  void StartQuery(const int task_id) {
    my_task = task_id;
    steps = (rand() % 5) + 1;

    // gt[task_id] = steps;

    state = ExecutionState::kWorking;
  }

  void Resume() {
    --steps;
    // ++result[my_task];

    if (steps <= 0) {
      state = ExecutionState::kFinished;
    }
  }

  ExecutionState state;
  int tid;

  int my_task;
  int steps;
};

// class NnExecutor {
//  public:
//   NnExecutor() : task_(), state_(ExecutionState::kFinished), cur_(nullptr) {
//     stack_.reserve(16);
//   }

//   void StartQuery(const Task& task) {
//     task_ = task;
//     stack_.clear();
//     cur_ = nullptr;
//     GetReductionResult(0, task.query_idx, &cached_result_addr_);
//     Execute();
//   }

//   void Init(const int tid) { tid_ = tid; }

//   void Resume() { Execute(); }

//   _NODISCARD bool Finished() const {
//     return state_ == ExecutionState::kFinished;
//   }

//  private:
//   void Execute() {
//     constexpr auto kernel_func = kernel::MyFunctor();
//     if (state_ == ExecutionState::kWorking) goto my_resume_point;

//     state_ = ExecutionState::kWorking;
//     cur_ = tree_ref->root_;

//     // Begin Iteration
//     while (cur_ != nullptr || !stack_.empty()) {
//       // Traverse all the way to left most leaf node
//       while (cur_ != nullptr) {
//         if (cur_->IsLeaf()) {
//           ReduceLeafNodeWithTask(0, cur_->uid, &task_);

//           // **** Coroutine Reuturn ****
//           return;
//         my_resume_point:
//           // ****************************

//           cur_ = nullptr;
//           continue;
//         }

//         // **** Reduction at tree node ****

//         const unsigned accessor_idx =
//             tree_ref->v_acc_[cur_->node_type.tree.idx_mid];

//         const float dist =
//             kernel_func(tree_ref->data_set_[accessor_idx],
//             task_.query_point);

//         *cached_result_addr_ = std::min(*cached_result_addr_, dist);

//         // **********************************

//         const int axis = cur_->node_type.tree.axis;
//         const float train = tree_ref->data_set_[accessor_idx].data[axis];
//         const kdt::Dir dir = task_.query_point.data[axis] < train
//                                  ? kdt::Dir::kLeft
//                                  : kdt::Dir::kRight;

//         stack_.push_back({cur_, axis, train, dir});
//         cur_ = cur_->GetChild(dir);
//       }

//       // We resume back from Break point, and now we are still in the branch
//       // node, we can check if there's any thing left on the stack.
//       if (!stack_.empty()) {
//         const auto [last_cur, axis, train, dir] = stack_.back();
//         stack_.pop_back();

//         // Check if there is a possibility of the NN lies on the other half
//         // If the difference between the query point and the other splitting
//         // plane is greater than the current found minimum distance, then it
//         is
//         // impossible to have a NN there.

//         Point4F a{};
//         Point4F b{};
//         a.data[axis] = task_.query_point.data[axis];
//         b.data[axis] = train;
//         const auto diff = kernel_func(a, b);
//         if (diff < *cached_result_addr_) {
//           cur_ = last_cur->GetChild(FlipDir(dir));
//         }
//       }
//     }

//     // Done traversals, Write back to final results
//     state_ = ExecutionState::kFinished;
//   }

//   int tid_;

//   // Actually essential data in a executor
//   int task_;
//   std::vector<CallStackField> stack_;
//   ExecutionState state_;
//   kdt::Node* cur_;

//   float* cached_result_addr_;  // a pointer to the USM of 1 float
// };