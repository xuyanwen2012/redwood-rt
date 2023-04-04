#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "../../LoadFile.hpp"
#include "../Executor.hpp"
#include "../ReducerHandler.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood.hpp"
#include "Redwood/Point.hpp"

using Task = std::pair<int, Point4F>;

_NODISCARD inline Point4F RandPoint() {
  Point4F p;
  p.data[0] = MyRand(0, 1024);
  p.data[1] = MyRand(0, 1024);
  p.data[2] = MyRand(0, 1024);
  p.data[3] = MyRand(0, 1024);
  return p;
}

float ComputeGroundTruth(const std::vector<Point4F>& data, Point4F q) {
  constexpr dist::Euclidean functor;

  auto my_min = std::numeric_limits<float>::infinity();
  for (int i = 0; i < data.size(); ++i) {
    const auto dist = functor(data[i], q);
    my_min = std::min(my_min, dist);
  }

  return my_min;
}

class QueryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const auto m = 1024;
    const auto num_threads = 1;

    in_data = load_data_from_file<Point4F>("../../../data/1m_nn_normal_4f.dat");
    const auto n = in_data.size();

    // For each thread
    q_data.resize(num_threads);
    const auto tasks_per_thread = m / num_threads;
    for (int tid = 0; tid < num_threads; ++tid) {
      for (int i = 0; i < tasks_per_thread; ++i) {
        q_data[tid].emplace(i, RandPoint());
      }
    }

    q2_data = q_data;
  }

  std::vector<std::queue<Task>> q_data;
  std::vector<std::queue<Task>> q2_data;

  std::vector<Point4F> in_data;
};

TEST_F(QueryTest, BuildQueue) {
  const auto n = q_data[0].size();
  EXPECT_EQ(n, 1024);
}

TEST_F(QueryTest, CpuTraversal) {
  const auto tid = 0;
  Executor<dist::Euclidean> cpu_exe{tid, 0, 0};
  const auto task = q_data[tid].front();

  cpu_exe.SetQuery(task);
  const auto my_result = cpu_exe.CpuTraverse();

  const auto gt = ComputeGroundTruth(in_data, task.second);

  EXPECT_EQ(my_result, gt);
}
