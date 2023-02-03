#pragma once

#include <algorithm>
#include <cassert>
#include <memory>
#include <vector>

#include "../KDTree.hpp"

// Reference to the kd tree
inline std::shared_ptr<kdt::KdTree> tree_ref;

enum class ExecutionState { kWorking, kFinished };

// Basically a nicer looking wrapper for passing function arguments
struct TaskList {
  TaskList(const Point4F* query_points, const int* query_indexis,
           const int num_query)
      : q_point(query_points), q_idx(query_indexis), m(num_query){};

  bool Done() const { return cur == m; }

  const Point4F* q_point;
  const int* q_idx;
  const int m;
  int cur;
};

template <typename ExecutorT>
inline void ProcessExecutors(std::vector<ExecutorT>& executors,
                             TaskList& task_list,
                             typename std::vector<ExecutorT>::iterator start,
                             typename std::vector<ExecutorT>::iterator& end) {
  for (auto it = start; it != end; ++it) {
    if (it->Finished()) {
      if (task_list.Done()) {
        it = executors.erase(it);
        --end;
        --it;
      } else {
        it->StartQuery(task_list.q_idx[task_list.cur]);
        ++task_list.cur;
      }
    } else {
      it->Resume();
    }
  }
}

class NnExecutor;

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
                  const Point4F* query_points,
                  const int* query_idx,  // basically uid for each query
                  const int my_m, const int num_batches, const int tid = 0)
      : tid_(tid),
        tasks_list_(query_points, query_idx, my_m),
        executors_(2 * num_batches, tid),
        num_batches_(num_batches) {
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

    // 'mid_point' will be modified
    auto mid_point = executors_.begin() + executors_.size() / 2;
    ProcessExecutors<DummyExecutor>(executors_, tasks_list_, executors_.begin(),
                                    mid_point);

    auto end_point = executors_.end();
    ProcessExecutors<DummyExecutor>(executors_, tasks_list_, mid_point,
                                    end_point);
  }

 private:
  int tid_;

  TaskList tasks_list_;
  std::vector<DummyExecutor> executors_;

  const int num_batches_;
};
