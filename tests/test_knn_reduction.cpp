// KNN leaf reduction: merge a leaf's distances into a sorted K-nearest set.
//   1. KnnSetMatchesBruteForce - host check of the KnnSet<float,K> structure.
//   2. KnnKernel<Point4F,K,F> through the redwood:: API vs a brute-force sorted
//      top-K, across distance metrics and edge cases. Runs on whichever backend
//      is linked, so CPU == CUDA == SYCL == ground truth.
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

template <class F>
void CheckKnnKernel(int num_active, int max_leaf, int num_leaves, F functor) {
  redwood::Init(1);
  std::mt19937 rng(13);

  const std::size_t lnt_n = std::size_t(num_leaves) * max_leaf;
  auto* lnt = redwood::UsmMalloc<Point4F>(lnt_n);
  for (std::size_t i = 0; i < lnt_n; ++i) lnt[i] = RandPoint(rng);

  auto* q = redwood::UsmMalloc<Point4F>(num_active);
  auto* nidx = redwood::UsmMalloc<int>(num_active);
  auto* out = redwood::UsmMalloc<float>(std::size_t(num_active) * kK);
  std::uniform_int_distribution<int> pick(0, num_leaves - 1);
  for (int i = 0; i < num_active; ++i) {
    q[i] = RandPoint(rng);
    nidx[i] = pick(rng);
    for (int t = 0; t < kK; ++t)
      out[i * kK + t] = std::numeric_limits<float>::max();
  }

  redwood::KnnKernel<Point4F, kK>(0, 0, lnt, max_leaf, q, nidx, num_active, out,
                                  functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < num_active; ++i) {
    const int leaf = nidx[i];
    std::vector<float> all(max_leaf);
    for (int j = 0; j < max_leaf; ++j)
      all[j] = functor(lnt[leaf * max_leaf + j], q[i]);
    std::sort(all.begin(), all.end());
    for (int t = 0; t < kK; ++t) {
      // If the leaf has fewer than K points, the remaining slots stay +inf.
      const float expected =
          t < max_leaf ? all[t] : std::numeric_limits<float>::max();
      EXPECT_FLOAT_EQ(out[i * kK + t], expected) << "slot " << i << " rank " << t;
    }
  }

  redwood::UsmFree(lnt);
  redwood::UsmFree(q);
  redwood::UsmFree(nidx);
  redwood::UsmFree(out);
}

}  // namespace

TEST(KnnReduction, KnnSetMatchesBruteForceKthNearest) {
  constexpr int kNumPoints = 500;
  constexpr dist::Euclidean functor;
  std::mt19937 rng(2024);
  std::vector<Point4F> points(kNumPoints);
  for (auto& p : points) p = RandPoint(rng);
  for (int trial = 0; trial < 20; ++trial) {
    const Point4F query = RandPoint(rng);
    KnnSet<float, kK> set;
    set.Reset();
    std::vector<float> all;
    for (const auto& p : points) {
      const float d = functor(p, query);
      set.Insert(d);
      all.push_back(d);
    }
    std::nth_element(all.begin(), all.begin() + (kK - 1), all.end());
    EXPECT_FLOAT_EQ(set.WorstDist(), all[kK - 1]) << "trial " << trial;
  }
}

TEST(KnnKernelTest, Euclidean) { CheckKnnKernel(256, 64, 16, dist::Euclidean{}); }
TEST(KnnKernelTest, Manhattan) { CheckKnnKernel(256, 64, 16, dist::Manhattan{}); }
TEST(KnnKernelTest, Chebyshev) { CheckKnnKernel(256, 64, 16, dist::Chebyshev{}); }

// Edge cases.
TEST(KnnKernelTest, LeafSmallerThanK) {
  // max_leaf (16) < K (32): only 16 real neighbors; the rest must stay +inf.
  CheckKnnKernel(256, 16, 16, dist::Euclidean{});
}
TEST(KnnKernelTest, NumActiveNotMultipleOfBlock) {
  CheckKnnKernel(1000, 64, 16, dist::Euclidean{});
}
