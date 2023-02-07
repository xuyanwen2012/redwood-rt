#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>

#include "../Kernel.hpp"

constexpr auto kNumStreams = 2;
extern sycl::device device;
extern sycl::context ctx;
extern sycl::queue qs[kNumStreams];
extern const Point4F* usm_leaf_node_table;

inline static auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

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

namespace redwood::internal {

void ProcessBhBuffer(const Point3F query_point, const Point4F* leaf_node_table,
                     const int* leaf_idx, int num_leaf_collected,
                     const Point4F* branch_data, int num_branch_collected,
                     Point3F* out, int leaf_max_size, int stream_id) {
  Point3F acc{};

  for (int i = 0; i < num_leaf_collected; ++i) {
    const auto leaf_id = leaf_idx[i];
    for (int j = 0; j < leaf_max_size; ++j) {
      const auto& p = leaf_node_table[leaf_id * leaf_max_size + j];
      acc += KernelFunc(p, query_point);
    }
  };

  if (branch_data != nullptr) {
    for (int i = 0; i < num_branch_collected; ++i) {
      const auto& p = branch_data[i];
      acc += KernelFunc(p, query_point);
    }
  }

  out[0] += acc;
}

}  // namespace redwood::internal