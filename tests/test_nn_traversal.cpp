// End-to-end NN traversal correctness test.
//
// This is the migrated, working replacement for examples/nn/tests/Query.cpp,
// which never ran: it built no kd-tree (left the global tree_ref null) and
// linked no backend. Here we drive the *real* production path -- build a
// kdt::KdTree, load its payload into backend USM, and run the actual
// Executor<>::CpuTraverse() pruning traversal -- then check every query against
// an independent brute-force nearest neighbor.
//
// It exercises real code: KdTree build + LoadPayload, the iterative/recursive
// traversal, KnnSet, the Euclidean functor, and the selected backend's
// UsmMalloc. Runs against any backend (-DREDWOOD_BACKEND=...).
#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <vector>

#include "../examples/nn/Executor.hpp"  // pulls KDTree, GlobalVars, ReducerHandler
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

class NnTraversal : public ::testing::Test {
 protected:
  static constexpr int kNumPoints = 4000;
  static constexpr int kLeafSize = 32;
  static constexpr int kBatch = 1024;

  void SetUp() override {
    std::mt19937 rng(20260615);
    data_.resize(kNumPoints);
    for (auto& p : data_) p = RandPoint(rng);

    const kdt::KdtParams params{kLeafSize};
    tree_ref = std::make_shared<kdt::KdTree>(params, data_.data(), kNumPoints);

    // Initialize the backend BEFORE any UsmMalloc. The SYCL backend needs its
    // device/context created first (malloc_shared binds to them); allocating
    // earlier binds the memory to an empty context and later frees fail with
    // UR_RESULT_ERROR_INVALID_VALUE. CUDA tolerates the reversed order via its
    // implicit context, which is why examples/nn/Main.cpp (AllocateLnt before
    // Init) happens to work on CUDA but is a latent cross-backend bug.
    rdc::Init(/*num_thread=*/1, /*batch_size=*/kBatch);

    const int num_leaf = tree_ref->GetStats().num_leaf_nodes;
    auto* lnt = rdc::AllocateLnt(num_leaf, kLeafSize);
    tree_ref->LoadPayload(lnt);

    final_results1.resize(kBatch);
  }

  void TearDown() override {
    rdc::Release();
    tree_ref.reset();
  }

  std::vector<Point4F> data_;
};

TEST_F(NnTraversal, CpuTraverseMatchesBruteForce) {
  Executor<dist::Euclidean> exe(/*tid=*/0, /*stream_id=*/0, /*uid=*/0);

  std::mt19937 rng(777);
  for (int t = 0; t < 50; ++t) {
    const Point4F q = RandPoint(rng);
    exe.SetQuery({0, q});
    const float got = exe.CpuTraverse();
    const float expected = BruteForceNN(data_, q);
    EXPECT_FLOAT_EQ(got, expected) << "query " << t;
  }
}
