// KNN leaf-node reduction: merge a leaf's distances into the running sorted
// K-nearest set.
//
// Two layers:
//   1. KnnSetMatchesBruteForce - host check of the KnnSet<float,K> data
//      structure against a brute-force k-th nearest.
//   2. KernelMatchesBruteForce - the real backend reduction through
//      redwood::KnnKernel, validated against a brute-force sorted top-K. Runs
//      on whichever backend is linked, so CPU == CUDA == SYCL == ground truth.
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "../examples/nn/KnnSet.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"

namespace {

constexpr int kK = 32;

Point4F RandPoint(std::mt19937& rng) {
  std::uniform_real_distribution<float> d(0.f, 1024.f);
  return Point4F{{d(rng), d(rng), d(rng), d(rng)}};
}

}  // namespace

TEST(KnnReduction, KnnSetMatchesBruteForceKthNearest) {
  constexpr int kNumPoints = 500;  // > K, so the set is fully populated
  constexpr dist::Euclidean functor;

  std::mt19937 rng(2024);
  std::vector<Point4F> points(kNumPoints);
  for (auto& p : points) p = RandPoint(rng);

  for (int trial = 0; trial < 20; ++trial) {
    const Point4F q = RandPoint(rng);
    KnnSet<float, kK> set;
    set.Reset();
    std::vector<float> all;
    for (const auto& p : points) {
      const float dist = functor(p, q);
      set.Insert(dist);
      all.push_back(dist);
    }
    std::nth_element(all.begin(), all.begin() + (kK - 1), all.end());
    EXPECT_FLOAT_EQ(set.WorstDist(), all[kK - 1]) << "trial " << trial;
  }
}

class KnnKernelTest : public ::testing::Test {
 protected:
  static constexpr int kMaxLeafSize = 64;  // >= K so a single leaf fills the set
  static constexpr int kNumLeaves = 16;
  static constexpr int kNumActive = 256;

  void SetUp() override {
    redwood::Init(/*num_threads=*/1);
    std::mt19937 rng(13);

    u_lnt = redwood::UsmMalloc<Point4F>(kNumLeaves * kMaxLeafSize);
    for (int i = 0; i < kNumLeaves * kMaxLeafSize; ++i) u_lnt[i] = RandPoint(rng);

    u_q = redwood::UsmMalloc<Point4F>(kNumActive);
    u_node_idx = redwood::UsmMalloc<int>(kNumActive);
    u_out = redwood::UsmMalloc<float>(kNumActive * kK);

    std::uniform_int_distribution<int> leaf_pick(0, kNumLeaves - 1);
    for (int i = 0; i < kNumActive; ++i) {
      u_q[i] = RandPoint(rng);
      u_node_idx[i] = leaf_pick(rng);
      for (int t = 0; t < kK; ++t)
        u_out[i * kK + t] = std::numeric_limits<float>::max();
    }
  }

  void TearDown() override {
    redwood::UsmFree(u_lnt);
    redwood::UsmFree(u_q);
    redwood::UsmFree(u_node_idx);
    redwood::UsmFree(u_out);
  }

  Point4F* u_lnt = nullptr;
  Point4F* u_q = nullptr;
  int* u_node_idx = nullptr;
  float* u_out = nullptr;
};

TEST_F(KnnKernelTest, MatchesBruteForceSortedTopK) {
  constexpr dist::Euclidean functor;
  redwood::KnnKernel<Point4F, kK>(/*tid=*/0, /*stream_id=*/0, u_lnt,
                                  kMaxLeafSize, u_q, u_node_idx, kNumActive,
                                  u_out, functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < kNumActive; ++i) {
    // Brute-force sorted distances for this slot's leaf.
    const int leaf = u_node_idx[i];
    std::vector<float> all(kMaxLeafSize);
    for (int j = 0; j < kMaxLeafSize; ++j)
      all[j] = functor(u_lnt[leaf * kMaxLeafSize + j], u_q[i]);
    std::sort(all.begin(), all.end());

    // The kernel keeps the K nearest in ascending order at u_out + i*K.
    for (int t = 0; t < kK; ++t)
      EXPECT_FLOAT_EQ(u_out[i * kK + t], all[t]) << "slot " << i << " rank " << t;
  }
}
