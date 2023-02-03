#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#include "../KDTree.hpp"
#include "HostKernelFunc.hpp"
#include "Redwood.hpp"

// Reference to the kd tree
inline std::shared_ptr<kdt::KdTree> tree_ref;

enum class ExecutionState { kWorking, kFinished };

// Basically a nicer looking wrapper for passing function arguments
struct TaskList {
  TaskList(const Point4F *query_points, const int *query_indexis,
           const int num_query)
      : q_point(query_points), q_idx(query_indexis), m(num_query) {
    cur = 0;
  };

  bool Done() const { return cur == m; }

  const Point4F *q_point;
  const int *q_idx;
  const int m;
  int cur;
};

template <typename ExecutorT>
inline void ProcessExecutors(std::vector<ExecutorT> &executors,
                             TaskList &task_list,
                             typename std::vector<ExecutorT>::iterator start,
                             typename std::vector<ExecutorT>::iterator &end) {
  for (auto it = start; it != end; ++it) {
    if (it->Finished()) {
      if (task_list.Done()) {
        it = executors.erase(it);
        --end;
        --it;
      } else {
        it->StartQuery(task_list.q_point[task_list.cur],
                       task_list.q_idx[task_list.cur]);
        ++task_list.cur;
      }
    } else {
      it->Resume();
    }
  }
}

struct CallStackField {
  kdt::Node *current;
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

class NnExecutor {
public:
  NnExecutor() = delete;
  NnExecutor(const int tid) : tid_(tid), state_(ExecutionState::kFinished) {
    stack_.reserve(16);
  }

  bool Finished() const { return state_ == ExecutionState::kFinished; }

  void StartQuery(const Point4F query_point, const int query_idx) {
    // if (query_idx < 32) std::cout << "StartQuery " << query_idx << std::endl;

    my_query_point_ = query_point;
    my_query_idx_ = query_idx;

    stack_.clear();
    cur_ = nullptr;

    // Basically, ask for a piece of USM from the device
    redwood::GetReductionResult(tid_, query_idx, &cached_result_addr_);
    Execute();
  }

  void Resume() { Execute(); }

private:
  // TODO: In future, this function will need to be code generated from our DSL
  void Execute() {
    constexpr auto kernel_func = MyFunctorHost();

    if (state_ == ExecutionState::kWorking)
      goto my_resume_point;

    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          // redwood::ReduceLeafNodeWithTask(0, cur_->uid, &task_);
          redwood::ReduceLeafNode(tid_, cur_->uid, my_query_idx_);

          // **** Coroutine Reuturn ****
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
            kernel_func(tree_ref->in_data_ref_[accessor_idx], my_query_point_);

        *cached_result_addr_ = std::min(*cached_result_addr_, dist);

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
        const auto diff = kernel_func(a, b);
        if (diff < *cached_result_addr_) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals, Write back to final results
    state_ = ExecutionState::kFinished;
  }

  int tid_;

  // Actually essential data in a executor
  // int task_;

  Point4F my_query_point_;
  int my_query_idx_;

  std::vector<CallStackField> stack_;

  ExecutionState state_;
  kdt::Node *cur_;

  float *cached_result_addr_; // a pointer to the USM of 1 float
};

// Each CPU thread should have one instance of a manager, each manager takes a
// subset of the query set (M). The purpose of Executor Manager is to orgranize
// the execution of each individual Traverser (see 'NNExecutor.hpp').
//
// For examples:
// 1) A CPU Sequential Executor Manager would just contain one traverser, and it
// process each sub task one at a time until all task are completed.
// 2) A Nearest Neighbor Executor would run one step of tree traversal for a
// batch of traversers (usally ~2000), and handles the coroutine 'suspend' and
// 'resume' etc.

// template <typename T, typename ExecutorT>
class ExecutorManager {
public:
  ExecutorManager() = delete;

  ExecutorManager(const std::shared_ptr<kdt::KdTree> tree,
                  const Point4F *query_points,
                  const int *query_idx, // basically uid for each query
                  const int my_m, const int num_batches, const int tid = 0)
      : tid_(tid), tasks_list_(query_points, query_idx, my_m),
        executors_(2 * num_batches, tid), num_batches_(num_batches) {
    // Save reference to
    if (!tree_ref) {
      std::cout << "[DEBUG] kdt::KdTree Reference Set!" << std::endl;
      tree_ref = tree;
    }

    assert(my_m % num_batches == 0);

    std::cout << "Manager (" << tid_ << "):\n"
              << "\tnum queries: " << my_m << '\n'
              << "\tnum batches: " << num_batches_ << '\n'
              << "\tnum executors: " << executors_.size() << '\n'
              << std::endl;
  }

  void StartTraversals() {
    std::cout << "Manager (" << tid_ << ") started.\n";

    while (!executors_.empty()) {
      // 'mid_point' will be modified
      auto mid_point = executors_.begin() + executors_.size() / 2;
      ProcessExecutors<NnExecutor>(executors_, tasks_list_, executors_.begin(),
                                   mid_point);

      redwood::rt::ExecuteCurrentBufferAsync(
          tid_, std::distance(executors_.begin(), mid_point));

      auto end_point = executors_.end();
      ProcessExecutors<NnExecutor>(executors_, tasks_list_, mid_point,
                                   end_point);

      redwood::rt::ExecuteCurrentBufferAsync(
          tid_, std::distance(mid_point, executors_.end()));
    }

    std::cout << "Manager (" << tid_ << ") has ended.\n";
  }

private:
  int tid_;

  TaskList tasks_list_;
  std::vector<NnExecutor> executors_;

  const int num_batches_;
};
