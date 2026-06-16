// Barnes-Hut leaf reduction: sum_j interaction(query, point_j) over a leaf.
//   1. GravityFunctorMatchesFormula - host check of dist::Gravity against a
//      double-precision reimplementation of the documented formula.
//   2. BarnesKernel<Point4F,F> through the redwood:: API vs a brute-force sum,
//      across interactions (Gravity/Gaussian/TopHat) and edge cases. Runs on
//      whichever backend is linked, so CPU == CUDA == SYCL == ground truth.
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

// Double-precision reimplementation of dist::Gravity (mass = first arg's [3]).
double GravityRef(const Point4F& a, const Point4F& b) {
  const double dx = double(a.data[0]) - b.data[0];
  const double dy = double(a.data[1]) - b.data[1];
  const double dz = double(a.data[2]) - b.data[2];
  const double dsq = dx * dx + dy * dy + dz * dz + 1e-9;
  const double inv = 1.0 / std::sqrt(dsq);
  const double inv3 = inv * inv * inv;
  const double wm = inv3 * double(a.data[3]);
  return dx * wm + dy * wm + dz * wm;
}

// Runs BarnesKernel and checks each slot's accumulated sum against a
// double-precision sum of the same functor (verifies the kernel's
// indexing/accumulation and that the interaction compiles in device code).
template <class F>
void CheckBarnesKernel(int num_active, int max_leaf, int num_leaves, F functor) {
  redwood::Init(1);
  std::mt19937 rng(2718);

  const std::size_t lnt_n = std::size_t(num_leaves) * max_leaf;
  auto* lnt = redwood::UsmMalloc<Point4F>(lnt_n);
  for (std::size_t i = 0; i < lnt_n; ++i) lnt[i] = RandBody(rng);

  auto* q = redwood::UsmMalloc<Point4F>(num_active);
  auto* nidx = redwood::UsmMalloc<int>(num_active);
  auto* out = redwood::UsmMalloc<float>(num_active);
  std::uniform_int_distribution<int> pick(0, num_leaves - 1);
  for (int i = 0; i < num_active; ++i) {
    q[i] = RandBody(rng);
    nidx[i] = pick(rng);
    out[i] = 0.f;
  }

  redwood::BarnesKernel(0, 0, lnt, max_leaf, q, nidx, num_active, out, functor);
  redwood::DeviceSynchronize();

  for (int i = 0; i < num_active; ++i) {
    const int leaf = nidx[i];
    double ref = 0.0;
    for (int j = 0; j < max_leaf; ++j)
      ref += double(functor(q[i], lnt[leaf * max_leaf + j]));
    EXPECT_NEAR(out[i], ref, 1e-3 * (std::fabs(ref) + 1.0)) << "slot " << i;
  }

  redwood::UsmFree(lnt);
  redwood::UsmFree(q);
  redwood::UsmFree(nidx);
  redwood::UsmFree(out);
}

}  // namespace

TEST(BhReduction, GravityFunctorMatchesFormula) {
  constexpr dist::Gravity functor;
  std::mt19937 rng(31415);
  for (int t = 0; t < 200; ++t) {
    const Point4F a = RandBody(rng), b = RandBody(rng);
    EXPECT_NEAR(functor(a, b), GravityRef(a, b),
                1e-4 * (std::fabs(GravityRef(a, b)) + 1.0));
  }
}

TEST(BhKernel, Gravity) { CheckBarnesKernel(256, 64, 16, dist::Gravity{}); }
TEST(BhKernel, Gaussian) { CheckBarnesKernel(256, 64, 16, dist::Gaussian{}); }
TEST(BhKernel, TopHat) { CheckBarnesKernel(256, 64, 16, dist::TopHat{}); }

// Edge cases.
TEST(BhKernel, NumActiveNotMultipleOfBlock) {
  CheckBarnesKernel(1000, 64, 16, dist::Gravity{});
}
TEST(BhKernel, FoldsIntoExistingResult) {
  // BarnesKernel accumulates (+=); seed a base and confirm it is added to.
  redwood::Init(1);
  constexpr int kN = 128, kLeaf = 64, kLeaves = 8;
  constexpr dist::Gravity functor;
  std::mt19937 rng(99);
  auto* lnt = redwood::UsmMalloc<Point4F>(kLeaves * kLeaf);
  for (int i = 0; i < kLeaves * kLeaf; ++i) lnt[i] = RandBody(rng);
  auto* q = redwood::UsmMalloc<Point4F>(kN);
  auto* nidx = redwood::UsmMalloc<int>(kN);
  auto* out = redwood::UsmMalloc<float>(kN);
  std::uniform_int_distribution<int> pick(0, kLeaves - 1);
  for (int i = 0; i < kN; ++i) {
    q[i] = RandBody(rng);
    nidx[i] = pick(rng);
    out[i] = 100.f;
  }
  redwood::BarnesKernel(0, 0, lnt, kLeaf, q, nidx, kN, out, functor);
  redwood::DeviceSynchronize();
  for (int i = 0; i < kN; ++i) {
    double ref = 100.0;
    for (int j = 0; j < kLeaf; ++j) ref += double(functor(q[i], lnt[nidx[i] * kLeaf + j]));
    EXPECT_NEAR(out[i], ref, 1e-3 * (std::fabs(ref) + 1.0)) << "slot " << i;
  }
  redwood::UsmFree(lnt);
  redwood::UsmFree(q);
  redwood::UsmFree(nidx);
  redwood::UsmFree(out);
}
