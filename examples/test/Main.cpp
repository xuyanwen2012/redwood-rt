#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

enum class ExecutionState { kWorking, kFinished };

std::vector<int> result;
std::vector<int> gt;

struct DummyExecutor {
  DummyExecutor() = delete;
  DummyExecutor(const int id) : state(ExecutionState::kFinished), tid(id) {}

  bool Finished() const { return state == ExecutionState::kFinished; }

  void StartQuery(const int task_id) {
    my_task = task_id;
    steps = (rand() % 5) + 1;

    gt[task_id] = steps;

    state = ExecutionState::kWorking;
  }

  void Resume() {
    --steps;
    ++result[my_task];

    if (steps <= 0) {
      state = ExecutionState::kFinished;
    }
  }

  ExecutionState state;
  int tid;

  int my_task;
  int steps;
};

template <typename ExecutorT>
inline void ProcessExecutors(std::vector<ExecutorT>& a, const int* tasks,
                             const int num_tasks, int& cur_task,
                             typename std::vector<ExecutorT>::iterator start,
                             typename std::vector<ExecutorT>::iterator& end) {
  for (auto it = start; it != end; ++it) {
    if (it->Finished()) {
      if (cur_task == num_tasks) {
        it = a.erase(it);
        --end;
        --it;
      } else {
        it->StartQuery(tasks[cur_task]);
        ++cur_task;
      }
    } else {
      it->Resume();
    }
  }
}

// template <typename ExecutorT>
// inline void ProcessExecutors(std::vector<ExecutorT>& a, std::vector<int>&
// tasks,
//                              typename std::vector<ExecutorT>::iterator start,
//                              typename std::vector<ExecutorT>::iterator end) {
//   for (auto it = start; it != end;) {
//     if (!it->Finished()) {
//       it->Resume();
//       ++it;
//       continue;
//     }

//     if (tasks.empty()) {
//       it = a.erase(it);
//       --end;
//     } else {
//       const auto task = tasks.back();
//       tasks.pop_back();
//       it->StartQuery(task);
//       ++it;
//     }
//   }
// }

struct Manager {
  Manager(const int n) : a(n, 1){};
  std::vector<DummyExecutor> a;
};

int main() {
  Manager mng(3);

  const auto num_batch = 32;
  const auto num_executors = 2 * num_batch;
  std::vector<DummyExecutor> a(num_executors, 999);

  constexpr auto num_tasks = 1024;
  result.resize(num_tasks);
  gt.resize(num_tasks);

  int* tasks = new int[num_tasks];
  for (int i = 0; i < num_tasks; ++i) {
    tasks[i] = i;
  }
  int cur_task = 0;

  srand(666);

  while (!a.empty()) {
    auto mid_point = a.begin() + a.size() / 2;
    ProcessExecutors<DummyExecutor>(a, tasks, num_tasks, cur_task, a.begin(),
                                    mid_point);

    // TODO: subject of BUG?
    // mid_point = a.begin() + a.size() / 2;
    auto end_point = a.end();
    ProcessExecutors<DummyExecutor>(a, tasks, num_tasks, cur_task, mid_point,
                                    end_point);
  }

  if (std::equal(result.begin(), result.end(), gt.begin())) {
    std::cout << "success" << std::endl;
  }

  return 0;
}