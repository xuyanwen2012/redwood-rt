// #include <gtest/gtest.h>

// #include "../KnnSet.hpp"  // Assuming the KnnSet code is in a file named
// KnnSet.hpp

// // Test fixture for KnnSet
// template <typename T, int K>
// class KnnSetTest : public ::testing::Test {
//  protected:
//   KnnSet<T, K> knn_set;
// };

// using KnnSetTypes = ::testing::Types<KnnSet<int, 3>>;

// TYPED_TEST_SUITE(KnnSetTest, KnnSetTypes);

// TYPED_TEST(KnnSetTest, InsertMaintainsOrder) {
//   this->knn_set.Insert(5);
//   this->knn_set.Insert(1);
//   this->knn_set.Insert(3);
//   EXPECT_EQ(this->knn_set.WorstDist(), 5);
// }

// // // Test cases for KnnSet with K > 1
// // using KnnSetTypes =
// //     ::testing::Types<KnnSet<int, 3>, KnnSet<float, 4>, KnnSet<double, 5>>;

// // TYPED_TEST_SUITE(KnnSetTest, KnnSetTypes);

// // TYPED_TEST(KnnSetTest, InsertMaintainsOrder) {
// //   this->knn_set.Insert(5);
// //   this->knn_set.Insert(1);
// //   this->knn_set.Insert(3);
// //   EXPECT_EQ(this->knn_set.WorstDist(), 5);
// // }

// // TYPED_TEST(KnnSetTest, Reset) {
// //   this->knn_set.Insert(5);
// //   this->knn_set.Insert(1);
// //   this->knn_set.Insert(3);
// //   this->knn_set.Reset();
// //   EXPECT_EQ(this->knn_set.WorstDist(),
// //             std::numeric_limits<typename TypeParam::value_type>::max());
// // }

// // // Test cases for KnnSet with K = 1
// // using KnnSetNNTypes =
// //     ::testing::Types<KnnSet<int, 1>, KnnSet<float, 1>, KnnSet<double, 1>>;
// // TYPED_TEST_SUITE(KnnSetTest, KnnSetNNTypes);

// // TYPED_TEST(KnnSetTest, InsertFindNN) {
// //   this->knn_set.Insert(5);
// //   this->knn_set.Insert(1);
// //   this->knn_set.Insert(3);
// //   EXPECT_EQ(this->knn_set.WorstDist(), 1);
// // }

// // TYPED_TEST(KnnSetTest, ResetNN) {
// //   this->knn_set.Insert(5);
// //   this->knn_set.Insert(1);
// //   this->knn_set.Insert(3);
// //   this->knn_set.Reset();
// //   EXPECT_EQ(this->knn_set.WorstDist(),
// //             std::numeric_limits<typename TypeParam::value_type>::max());
// // }
