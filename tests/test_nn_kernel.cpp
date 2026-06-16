// Backend-agnostic test of the nearest-neighbor leaf reduction through the
// redwood:: API. The same body runs against whichever backend is linked
// (-DREDWOOD_BACKEND=cpu|cuda|sycl), so CPU == CUDA == SYCL == ground truth.
//
// Covers all three NN distance metrics (the kernel runs differently per functor
// and must compile in device code), plus edge cases that a naive/buggy kernel
// would get wrong (batch size not a multiple of the block, single point).
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>

#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"

namespace {

Point4F RandPoint(std::mt19937& rng) {
  std::uniform_real_distribution<float> d(0.f, 1024.f);
  return Point4F{{d(rng), d(rng), d(rng), d(rng)}};
}

// Independent brute-force min over a leaf's points, using the same functor the
// kernel uses (so the test isolates the kernel's reduction/indexing, and cross-
// backend equivalence is the real check).
template <class F>
float BruteForceMin(const Point4F* lnt, int leaf_id, int max_leaf_size,
                    Point4F q, F functor) {
  auto best = std::numeric_limits<float>::max();
  for (int j = 0; j < max_leaf_size; ++j)
    best = std::min(best, functor(lnt[leaf_id * max_leaf_size + j], q));
  return best;
}

template <class F>
void CheckNnKernel(int num_active, int max_leaf, int num_leaves, F functor) {
  redwood::Init(1);
  std::mt19937 rng(114514);

  const std::size_t lnt_n = std::size_t(num_leaves) * max_leaf;
  auto* lnt = redwood::UsmMalloc<Point4F>(lnt_n);
  for (std::size_t i = 0; i < lnt_n; ++i) lnt[i] = RandPoint(rng);

  auto* q = redwood::UsmMalloc<Point4F>(num_active);
  auto* nidx = redwood::UsmMalloc<int>(num_active);
  auto* out = redwood::UsmMalloc<float>(num_active);
  std::uniform_int_distribution<int> pick(0, num_leaves - 1);
  for (int i = 0; i < num_active; ++i) {
    q[i] = RandPoint(rng);
    nidx[i] = pick(rng);
    out[i] = std::numeric_limits<float>::max();
  }

  redwood::NearestNeighborKernel(0, 0, lnt, max_leaf, q, nidx, num_active, out,
                                 functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < num_active; ++i)
    EXPECT_FLOAT_EQ(out[i], BruteForceMin(lnt, nidx[i], max_leaf, q[i], functor))
        << "slot " << i;

  redwood::UsmFree(lnt);
  redwood::UsmFree(q);
  redwood::UsmFree(nidx);
  redwood::UsmFree(out);
}

}  // namespace

TEST(NnKernel, Euclidean) { CheckNnKernel(1024, 32, 16, dist::Euclidean{}); }
TEST(NnKernel, Manhattan) { CheckNnKernel(1024, 32, 16, dist::Manhattan{}); }
TEST(NnKernel, Chebyshev) { CheckNnKernel(1024, 32, 16, dist::Chebyshev{}); }

// Edge cases.
TEST(NnKernel, NumActiveNotMultipleOfBlock) {
  CheckNnKernel(1000, 32, 16, dist::Euclidean{});  // grid boundary
}
TEST(NnKernel, SinglePointSingleLeaf) {
  CheckNnKernel(1, 1, 1, dist::Euclidean{});
}
TEST(NnKernel, LargeLeaf) {
  CheckNnKernel(512, 256, 8, dist::Euclidean{});
}

TEST(NnKernel, FoldsIntoExistingResult) {
  // Pre-seed u_out=0: the kernel keeps the running min, never making it worse.
  redwood::Init(1);
  constexpr int kN = 256, kLeaf = 32, kLeaves = 8;
  std::mt19937 rng(7);
  auto* lnt = redwood::UsmMalloc<Point4F>(kLeaves * kLeaf);
  for (int i = 0; i < kLeaves * kLeaf; ++i) lnt[i] = RandPoint(rng);
  auto* q = redwood::UsmMalloc<Point4F>(kN);
  auto* nidx = redwood::UsmMalloc<int>(kN);
  auto* out = redwood::UsmMalloc<float>(kN);
  std::uniform_int_distribution<int> pick(0, kLeaves - 1);
  for (int i = 0; i < kN; ++i) {
    q[i] = RandPoint(rng);
    nidx[i] = pick(rng);
    out[i] = 0.f;
  }
  redwood::NearestNeighborKernel(0, 0, lnt, kLeaf, q, nidx, kN, out,
                                 dist::Euclidean{});
  redwood::DeviceSynchronize();
  for (int i = 0; i < kN; ++i) EXPECT_FLOAT_EQ(out[i], 0.f);
  redwood::UsmFree(lnt);
  redwood::UsmFree(q);
  redwood::UsmFree(nidx);
  redwood::UsmFree(out);
}
