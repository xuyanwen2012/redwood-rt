// Unit tests for the shared Point<Dim, T> vector type (include/Redwood/Point.hpp).
// Pure value type, no backend involved.
#include "Redwood/Point.hpp"

#include <gtest/gtest.h>

TEST(Point, AddSubtract) {
  Point4F a{{1.f, 2.f, 3.f, 4.f}};
  Point4F b{{4.f, 3.f, 2.f, 1.f}};

  const auto sum = a + b;
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(sum.data[i], 5.f);

  const auto diff = a - b;
  EXPECT_FLOAT_EQ(diff.data[0], -3.f);
  EXPECT_FLOAT_EQ(diff.data[3], 3.f);
}

TEST(Point, ScalarMulDiv) {
  Point4F a{{2.f, 4.f, 6.f, 8.f}};

  const auto scaled = a * 2.f;
  EXPECT_FLOAT_EQ(scaled.data[0], 4.f);
  EXPECT_FLOAT_EQ(scaled.data[3], 16.f);

  const auto halved = a / 2.f;
  EXPECT_FLOAT_EQ(halved.data[0], 1.f);
  EXPECT_FLOAT_EQ(halved.data[3], 4.f);
}

TEST(Point, CompoundAssignAndEquality) {
  Point4F a{{1.f, 1.f, 1.f, 1.f}};
  Point4F b{{1.f, 2.f, 3.f, 4.f}};
  a += b;

  EXPECT_EQ(a, (Point4F{{2.f, 3.f, 4.f, 5.f}}));
  EXPECT_NE(a, b);
}

TEST(Point, DimensionsAndTypes) {
  // Sanity: the type aliases have the expected dimensionality.
  EXPECT_EQ(sizeof(Point2F) / sizeof(float), 2u);
  EXPECT_EQ(sizeof(Point3F) / sizeof(float), 3u);
  EXPECT_EQ(sizeof(Point4F) / sizeof(float), 4u);
}
