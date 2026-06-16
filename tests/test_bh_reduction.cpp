// Barnes-Hut leaf-node reduction correctness (the computation in
// examples/barnes/ReducerHandler.hpp::ReduceLeafNode): accumulate
// sum_j Gravity(query, point_j) over a leaf's points.
//
// We validate two things against an independent double-precision reference:
//   1. the dist::Gravity functor evaluates the documented force formula, and
//   2. the leaf reduction is the linear sum of those per-point interactions.
//
// NOTE: like KNN, BH has no kernel in the redwood:: backend API (the GPU launch
// in barnes/ReducerHandler.hpp is commented out), so this is a CPU
// algorithm-correctness test, not yet cross-backend. When a BarnesKernel is
// added it can be promoted to a CPU == CUDA == ground-truth check.
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "Functors/DistanceMetrics.hpp"

namespace {

Point4F RandBody(std::mt19937& rng) {
  std::uniform_real_distribution<float> pos(0.f, 1024.f);
  std::uniform_real_distribution<float> mass(0.1f, 10.f);
  return Point4F{{pos(rng), pos(rng), pos(rng), mass(rng)}};
}

// Independent, double-precision reimplementation of dist::Gravity.
// Mirrors the functor's argument roles: mass is taken from the first argument
// (`a.data[3]`), matching dist::Gravity in DistanceMetrics.hpp.
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

    // float math vs double reference: bound by relative precision.
    EXPECT_NEAR(got, ref, 1e-4 * (std::fabs(ref) + 1.0)) << "sample " << t;
  }
}

TEST(BhReduction, LeafSumMatchesReference) {
  constexpr dist::Gravity functor;
  constexpr int kLeafSize = 64;

  std::mt19937 rng(27182);
  std::vector<Point4F> leaf(kLeafSize);
  for (auto& b : leaf) b = RandBody(rng);

  for (int trial = 0; trial < 20; ++trial) {
    const Point4F q = RandBody(rng);

    // The reduction under test: sum of per-point gravity interactions.
    float my_sum = 0.f;
    for (const auto& p : leaf) my_sum += functor(q, p);

    // Independent reference accumulated in double precision.
    double ref_sum = 0.0;
    for (const auto& p : leaf) ref_sum += GravityRef(q, p);

    EXPECT_NEAR(my_sum, ref_sum, 1e-3 * (std::fabs(ref_sum) + 1.0))
        << "trial " << trial;
  }
}
