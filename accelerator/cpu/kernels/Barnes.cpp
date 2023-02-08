
#include <algorithm>
#include <cmath>

#include "PointCloud.hpp"
#include "accelerator/Kernels.hpp"

inline auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

inline Point3F KernelFunc(const Point4F p, const Point3F q) {
  const auto dx = p.data[0] - q.data[0];
  const auto dy = p.data[1] - q.data[1];
  const auto dz = p.data[2] - q.data[2];
  const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
  const auto inv_dist = rsqrtf(dist_sqr);
  const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
  const auto with_mass = inv_dist3 * p.data[3];
  return {dx * with_mass, dy * with_mass, dz * with_mass};
}

// Naive
inline Point3F SumLeaf(const Point3F q_point, const Point4F* base,
                       const int leaf_max_size) {
  Point3F acc{};
  for (int i = 0; i < leaf_max_size; ++i) acc += KernelFunc(base[i], q_point);
  return acc;
}

namespace redwood::accelerator {

void LaunchBhKernel(const Point3F query_point, const int q_idx,
                    const Point4F* leaf_node_table, const int* leaf_idx,
                    const int num_leaf_collected, const Point4F* branch_data,
                    const int num_branch_collected, Point3F* out,
                    const int leaf_max_size, const int stream_id) {
  for (int i = 0; i < num_leaf_collected; ++i) {
    const auto leaf_id = leaf_idx[i];

    auto acc = SumLeaf(query_point, leaf_node_table + leaf_id * leaf_max_size,
                       leaf_max_size);

    out[q_idx] += acc;
  }
}

}  // namespace redwood::accelerator