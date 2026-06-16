// Unit tests for KnnSet (examples/nn/KnnSet.hpp): the bounded result set that
// accumulates the k smallest distances during a traversal.
//
// KnnSet<T,1> is the NN specialization (a single running min); KnnSet<T,K>
// keeps the K smallest values sorted ascending, with WorstDist() == the k-th
// nearest. These are the invariants the traversal's pruning test relies on.
#include "../examples/nn/KnnSet.hpp"

#include <gtest/gtest.h>

#include <limits>

TEST(KnnSetNN, ResetIsInfinite) {
  KnnSet<float, 1> s;
  s.Reset();
  EXPECT_EQ(s.WorstDist(), std::numeric_limits<float>::max());
}

TEST(KnnSetNN, KeepsMinimum) {
  KnnSet<float, 1> s;
  s.Reset();
  for (float v : {5.f, 3.f, 9.f, 1.f, 4.f}) s.Insert(v);
  EXPECT_FLOAT_EQ(s.WorstDist(), 1.f);
}

TEST(KnnSetK, KeepsKSmallestSorted) {
  KnnSet<float, 3> s;
  s.Reset();
  for (float v : {5.f, 3.f, 1.f, 4.f, 2.f}) s.Insert(v);
  // Smallest three are {1,2,3}; the "worst" kept (3rd nearest) is 3.
  EXPECT_FLOAT_EQ(s.WorstDist(), 3.f);
}

TEST(KnnSetK, FewerInsertsThanK) {
  KnnSet<float, 3> s;
  s.Reset();
  s.Insert(7.f);
  s.Insert(2.f);
  // Only two real values inserted; the third slot stays at +inf.
  EXPECT_EQ(s.WorstDist(), std::numeric_limits<float>::max());
}
