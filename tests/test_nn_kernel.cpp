// Backend-agnostic test of the nearest-neighbor leaf-node reduction through
// the public redwood:: API.
//
// It exercises the real backend entry points an application uses --
// redwood::Init, redwood::UsmMalloc, redwood::NearestNeighborKernel,
// redwood::DeviceSynchronize -- and checks the kernel against an independent
// brute-force ground truth. Linking decides which backend runs: reconfigure
// with -DREDWOOD_BACKEND=cpu|cuda|sycl and the same assertions validate each.
//
// Sizes are chosen to match the CUDA warp kernel's designed granularity (a full
// 1024-wide batch); the CPU backend handles any size.
#include <gtest/gtest.h>

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

// Brute-force minimum distance from q to the points of leaf `leaf_id`,
// using the same functor the kernel uses (so SOFTENING etc. match exactly).
float BruteForceMin(const Point4F* lnt, int leaf_id, int max_leaf_size,
                    Point4F q) {
  constexpr dist::Euclidean functor;
  auto best = std::numeric_limits<float>::max();
  for (int j = 0; j < max_leaf_size; ++j) {
    best = std::min(best, functor(lnt[leaf_id * max_leaf_size + j], q));
  }
  return best;
}

}  // namespace

class NnKernel : public ::testing::Test {
 protected:
  static constexpr int kMaxLeafSize = 32;
  static constexpr int kNumLeaves = 16;
  static constexpr int kNumActive = 1024;

  void SetUp() override {
    redwood::Init(/*num_threads=*/1);
    std::mt19937 rng(114514);

    u_lnt = redwood::UsmMalloc<Point4F>(kNumLeaves * kMaxLeafSize);
    for (int i = 0; i < kNumLeaves * kMaxLeafSize; ++i) u_lnt[i] = RandPoint(rng);

    u_q = redwood::UsmMalloc<Point4F>(kNumActive);
    u_node_idx = redwood::UsmMalloc<int>(kNumActive);
    u_out = redwood::UsmMalloc<float>(kNumActive);

    std::uniform_int_distribution<int> leaf_pick(0, kNumLeaves - 1);
    for (int i = 0; i < kNumActive; ++i) {
      u_q[i] = RandPoint(rng);
      u_node_idx[i] = leaf_pick(rng);
      u_out[i] = std::numeric_limits<float>::max();
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

TEST_F(NnKernel, MatchesBruteForce) {
  constexpr dist::Euclidean functor;
  redwood::NearestNeighborKernel(/*tid=*/0, /*stream_id=*/0, u_lnt, kMaxLeafSize,
                                 u_q, u_node_idx, kNumActive, u_out, functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < kNumActive; ++i) {
    const float expected =
        BruteForceMin(u_lnt, u_node_idx[i], kMaxLeafSize, u_q[i]);
    EXPECT_FLOAT_EQ(u_out[i], expected) << "query " << i;
  }
}

TEST_F(NnKernel, FoldsIntoExistingResult) {
  // Pre-seed u_out with the best possible value: the kernel must keep the
  // running min, i.e. never make a result worse. This is what lets the runtime
  // accumulate across multiple leaf-node visits for the same query.
  for (int i = 0; i < kNumActive; ++i) u_out[i] = 0.f;

  constexpr dist::Euclidean functor;
  redwood::NearestNeighborKernel(0, 0, u_lnt, kMaxLeafSize, u_q, u_node_idx,
                                 kNumActive, u_out, functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < kNumActive; ++i) EXPECT_FLOAT_EQ(u_out[i], 0.f);
}
