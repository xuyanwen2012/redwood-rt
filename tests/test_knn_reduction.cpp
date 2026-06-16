// KNN leaf-node reduction correctness (the computation in examples/knn's
// DebugCpuReduction): insert every leaf point's distance into a KnnSet<float,K>
// and the kept set must be the K smallest distances.
//
// NOTE: this is a CPU algorithm-correctness test, not yet a cross-backend one.
// KNN has no kernel in the redwood:: backend API (the GPU launch in
// examples/knn/ReducerHandler.hpp is commented out), so there is nothing to
// compare a CPU result against on CUDA/SYCL. Once a KnnKernel is added to the
// backend API this test can be lifted to run through redwood::KnnKernel like
// test_nn_kernel does, and validate CPU == CUDA == ground truth.
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "../examples/nn/KnnSet.hpp"
#include "Functors/DistanceMetrics.hpp"

namespace {

Point4F RandPoint(std::mt19937& rng) {
  std::uniform_real_distribution<float> d(0.f, 1024.f);
  return Point4F{{d(rng), d(rng), d(rng), d(rng)}};
}

}  // namespace

TEST(KnnReduction, MatchesBruteForceKthNearest) {
  constexpr int kK = 32;
  constexpr int kNumPoints = 500;  // > K, so the set is fully populated
  constexpr dist::Euclidean functor;

  std::mt19937 rng(2024);
  std::vector<Point4F> points(kNumPoints);
  for (auto& p : points) p = RandPoint(rng);

  for (int trial = 0; trial < 20; ++trial) {
    const Point4F q = RandPoint(rng);

    // The reduction under test: fold all distances through the KnnSet.
    KnnSet<float, kK> set;
    set.Reset();
    std::vector<float> all;
    all.reserve(kNumPoints);
    for (const auto& p : points) {
      const float dist = functor(p, q);
      set.Insert(dist);
      all.push_back(dist);
    }

    // Independent ground truth: the K-th smallest distance.
    std::nth_element(all.begin(), all.begin() + (kK - 1), all.end());
    const float kth_smallest = all[kK - 1];

    // WorstDist() is the k-th nearest (largest of the kept K).
    EXPECT_FLOAT_EQ(set.WorstDist(), kth_smallest) << "trial " << trial;
  }
}

TEST(KnnReduction, FewerPointsThanKLeavesInfinity) {
  constexpr int kK = 32;
  constexpr dist::Euclidean functor;
  const Point4F q{{1.f, 2.f, 3.f, 4.f}};

  KnnSet<float, kK> set;
  set.Reset();
  // Insert only 5 points (< K): the k-th slot must stay at +inf.
  std::mt19937 rng(9);
  for (int i = 0; i < 5; ++i) set.Insert(functor(RandPoint(rng), q));

  EXPECT_EQ(set.WorstDist(), std::numeric_limits<float>::max());
}
