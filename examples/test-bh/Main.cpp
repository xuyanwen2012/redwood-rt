#include <iostream>
#include <random>

// Redwood related
#include "../../redwood/BhBuffer.hpp"
#include "PointCloud.hpp"
#include "Redwood.hpp"
#include "UsmAlloc.hpp"

float my_rand(float min = 0.0, float max = 1.0) {
  // 114514 and 233
  static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(generator);
}

namespace redwood {
namespace internal {
extern void ProcessBhBuffer(const Point3F query_point,
                            const Point4F* leaf_node_table, const int* leaf_idx,
                            int num_leaf_collected, const Point4F* branch_data,
                            int num_branch_collected, Point3F* out,
                            int leaf_max_size, int stream_id);
}
}  // namespace redwood

int main() {
  const auto n = 1024 * 32;
  const auto m = 1024;
  const auto batch_size = 1024;

  auto in_data = static_cast<Point4F*>(malloc(n * sizeof(Point4F)));
  auto q_data = static_cast<Point3F*>(malloc(m * sizeof(Point3F)));

  static auto rand_point4f = []() {
    return Point4F{
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
    };
  };

  static auto rand_point3f = []() {
    return Point3F{
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
    };
  };

  std::generate_n(in_data, n, rand_point4f);
  std::generate_n(q_data, m, rand_point3f);

  redwood::BhBuffer<Point4F, Point3F, Point3F> bh_pack;

  bh_pack.Allocate(batch_size);

  bh_pack.SetTask(q_data[0]);

  for (int i = 0; i < 256; ++i) {
    bh_pack.PushLeaf(i);
  }

  for (int i = 0; i < 256; ++i) {
    bh_pack.PushBranch(rand_point4f());
  }

  Point3F result{};
  // CpuProcessBhBuffer(bh_pack.my_query, bh_pack.LeafNodeData(),
  //                    bh_pack.NumLeafsCollected(), bh_pack.BranchNodeData(),
  //                    bh_pack.NumBranchCollected(), &result, 32, 0);

  redwood::internal::ProcessBhBuffer(
      bh_pack.my_query, in_data, bh_pack.LeafNodeData(),
      bh_pack.NumLeafsCollected(), nullptr, 0, &result, 32, 0);

  std::cout << result << std::endl;

  return EXIT_SUCCESS;
}