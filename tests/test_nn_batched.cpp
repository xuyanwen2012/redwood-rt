// End-to-end test of the BATCHED coroutine pipeline (the real heterogeneous
// traverse-compute path), not just the recursive CpuTraverse.
//
// It drives the same loop examples/nn/Main.cpp uses for the GPU path:
// Executor::StartQuery/Resume run the coroutine traversal, leaf hits are batched
// via rdc::ReduceLeafNode, rdc::LaunchAsyncWorkQueue fires the backend kernel,
// and the per-executor result slot feeds the pruning decision. The final
// per-query result (written into final_results1 when each Executor finishes)
// must equal a brute-force nearest neighbor.
#include <gtest/gtest.h>

#include <limits>
#include <queue>
#include <random>
#include <vector>

#include "../examples/nn/Executor.hpp"
#include "Functors/DistanceMetrics.hpp"

namespace {

Point4F RandPoint(std::mt19937& rng) {
  std::uniform_real_distribution<float> d(0.f, 1024.f);
  return Point4F{{d(rng), d(rng), d(rng), d(rng)}};
}

float BruteForceNN(const std::vector<Point4F>& data, Point4F q) {
  constexpr dist::Euclidean functor;
  auto best = std::numeric_limits<float>::max();
  for (const auto& p : data) best = std::min(best, functor(p, q));
  return best;
}

}  // namespace

TEST(NnBatched, PipelineMatchesBruteForce) {
  constexpr int kNumPoints = 2000;
  constexpr int kLeafSize = 16;
  constexpr int kBatch = 32;
  constexpr int kNumQueries = 256;

  std::mt19937 rng(2026);
  std::vector<Point4F> data(kNumPoints);
  for (auto& p : data) p = RandPoint(rng);

  const kdt::KdtParams params{kLeafSize};
  tree_ref = std::make_shared<kdt::KdTree>(params, data.data(), kNumPoints);

  rdc::Init(/*num_thread=*/1, /*batch_size=*/kBatch);
  const int num_leaf = tree_ref->GetStats().num_leaf_nodes;
  auto* lnt = rdc::AllocateLnt(num_leaf, kLeafSize);
  tree_ref->LoadPayload(lnt);
  final_results1.assign(kNumQueries, -1.f);

  // Queries, and an independent brute-force answer for each.
  std::vector<Point4F> queries(kNumQueries);
  std::vector<float> expected(kNumQueries);
  for (int i = 0; i < kNumQueries; ++i) {
    queries[i] = RandPoint(rng);
    expected[i] = BruteForceNN(data, queries[i]);
  }

  // One stream's worth of executors (uid = slot in the batch).
  std::vector<Executor<dist::Euclidean>> exes;
  exes.reserve(kBatch);
  for (int i = 0; i < kBatch; ++i) exes.emplace_back(0, 0, i);

  std::queue<std::pair<int, Point4F>> q;
  for (int i = 0; i < kNumQueries; ++i) q.emplace(i, queries[i]);

  auto flush = [] {
    rdc::LaunchAsyncWorkQueue(0, 0);
    redwood::DeviceStreamSynchronize(0, 0);
    rdc::ResetBuffer(0, 0);
  };

  // Main-style loop: keep slots full, batch leaf reductions, flush per pass.
  while (!q.empty()) {
    int it = 0;
    while (it < kBatch) {
      if (exes[it].Finished()) {
        if (!q.empty()) {
          exes[it].SetQuery(q.front());
          q.pop();
          exes[it].StartQuery();
        }
        ++it;
      } else {
        exes[it].Resume();
        if (!exes[it].Finished()) ++it;
      }
    }
    flush();
  }

  // Drain executors still in flight.
  bool busy = true;
  while (busy) {
    busy = false;
    for (int it = 0; it < kBatch; ++it)
      if (!exes[it].Finished()) {
        exes[it].Resume();
        busy = true;
      }
    flush();
  }

  int mismatches = 0;
  for (int i = 0; i < kNumQueries; ++i) {
    if (std::abs(final_results1[i] - expected[i]) > 1e-3f * (expected[i] + 1.f))
      ++mismatches;
    EXPECT_NEAR(final_results1[i], expected[i], 1e-3f * (expected[i] + 1.f))
        << "query " << i;
  }
  EXPECT_EQ(mismatches, 0) << mismatches << "/" << kNumQueries << " wrong";

  rdc::Release();
  tree_ref.reset();
}
