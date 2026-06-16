// Barnes-Hut leaf-node reduction: sum_j Gravity(query, point_j) over a leaf,
// folded into the running result.
//
// Two layers:
//   1. GravityFunctorMatchesFormula - host check of the dist::Gravity math
//      against a double-precision reference.
//   2. KernelMatchesBruteForce - the real backend reduction through
//      redwood::BarnesKernel, validated against a brute-force reference. This
//      runs on whichever backend is linked (-DREDWOOD_BACKEND=cpu|cuda|sycl),
//      so CPU == CUDA == SYCL == ground truth.
#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"

namespace {

Point4F RandBody(std::mt19937& rng) {
  std::uniform_real_distribution<float> pos(0.f, 1024.f);
  std::uniform_real_distribution<float> mass(0.1f, 10.f);
  return Point4F{{pos(rng), pos(rng), pos(rng), mass(rng)}};
}

// Independent, double-precision reimplementation of dist::Gravity. Mirrors the
// functor's argument roles: mass is taken from the first argument (a.data[3]),
// matching dist::Gravity and BarnesKernel's functor(query, point) call.
double GravityRef(const Point4F& a, const Point4F& b) {
  const double dx = static_cast<double>(a.data[0]) - b.data[0];
  const double dy = static_cast<double>(a.data[1]) - b.data[1];
  const double dz = static_cast<double>(a.data[2]) - b.data[2];
  const double dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9;
  const double inv_dist = 1.0 / std::sqrt(dist_sqr);
  const double inv_dist3 = inv_dist * inv_dist * inv_dist;
  const double with_mass = inv_dist3 * static_cast<double>(a.data[3]);
  return dx * with_mass + dy * with_mass + dz * with_mass;
}

}  // namespace

TEST(BhReduction, GravityFunctorMatchesFormula) {
  constexpr dist::Gravity functor;
  std::mt19937 rng(31415);
  for (int t = 0; t < 200; ++t) {
    const Point4F q = RandBody(rng);
    const Point4F p = RandBody(rng);
    const float got = functor(q, p);
    const double ref = GravityRef(q, p);
    EXPECT_NEAR(got, ref, 1e-4 * (std::fabs(ref) + 1.0)) << "sample " << t;
  }
}

class BhKernel : public ::testing::Test {
 protected:
  static constexpr int kMaxLeafSize = 64;
  static constexpr int kNumLeaves = 16;
  static constexpr int kNumActive = 256;

  void SetUp() override {
    redwood::Init(/*num_threads=*/1);
    std::mt19937 rng(2718);

    u_lnt = redwood::UsmMalloc<Point4F>(kNumLeaves * kMaxLeafSize);
    for (int i = 0; i < kNumLeaves * kMaxLeafSize; ++i) u_lnt[i] = RandBody(rng);

    u_q = redwood::UsmMalloc<Point4F>(kNumActive);
    u_node_idx = redwood::UsmMalloc<int>(kNumActive);
    u_out = redwood::UsmMalloc<float>(kNumActive);

    std::uniform_int_distribution<int> leaf_pick(0, kNumLeaves - 1);
    for (int i = 0; i < kNumActive; ++i) {
      u_q[i] = RandBody(rng);
      u_node_idx[i] = leaf_pick(rng);
      u_out[i] = 0.f;
    }
  }

  void TearDown() override {
    redwood::UsmFree(u_lnt);
    redwood::UsmFree(u_q);
    redwood::UsmFree(u_node_idx);
    redwood::UsmFree(u_out);
  }

  double RefSum(int slot) const {
    double s = 0.0;
    const int leaf = u_node_idx[slot];
    for (int j = 0; j < kMaxLeafSize; ++j) {
      s += GravityRef(u_q[slot], u_lnt[leaf * kMaxLeafSize + j]);
    }
    return s;
  }

  Point4F* u_lnt = nullptr;
  Point4F* u_q = nullptr;
  int* u_node_idx = nullptr;
  float* u_out = nullptr;
};

TEST_F(BhKernel, MatchesBruteForce) {
  constexpr dist::Gravity functor;
  redwood::BarnesKernel(/*tid=*/0, /*stream_id=*/0, u_lnt, kMaxLeafSize, u_q,
                        u_node_idx, kNumActive, u_out, functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < kNumActive; ++i) {
    const double ref = RefSum(i);
    EXPECT_NEAR(u_out[i], ref, 1e-3 * (std::fabs(ref) + 1.0)) << "slot " << i;
  }
}

TEST_F(BhKernel, FoldsIntoExistingResult) {
  // BarnesKernel accumulates (+=), so a pre-seeded base must be added to.
  for (int i = 0; i < kNumActive; ++i) u_out[i] = 100.f;

  constexpr dist::Gravity functor;
  redwood::BarnesKernel(0, 0, u_lnt, kMaxLeafSize, u_q, u_node_idx, kNumActive,
                        u_out, functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < kNumActive; ++i) {
    const double ref = 100.0 + RefSum(i);
    EXPECT_NEAR(u_out[i], ref, 1e-3 * (std::fabs(ref) + 1.0)) << "slot " << i;
  }
}
