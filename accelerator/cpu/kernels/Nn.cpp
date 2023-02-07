
#include <algorithm>
#include <cmath>

#include "accelerator/Kernels.hpp"

inline float KernelFunc(const Point4F p, const Point4F q) {
  auto dist = float();

  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

namespace redwood::accelerator {

void LaunchNnKernel(const Point4F* query_points, const Point4F* leaf_node_table,
                    const int* query_idx, const int* leaf_idx, float* out,
                    const int num, const int leaf_max_size,
                    const int stream_id) {
  auto my_query_points = static_cast<const Point4F*>(query_points);

  for (int batch_id = 0; batch_id < num; ++batch_id) {
    const auto leaf_id = leaf_idx[batch_id];
    const auto q_point = my_query_points[batch_id];
    const auto q_idx = query_idx[batch_id];

    auto my_min = std::numeric_limits<float>::max();
    for (int i = 0; i < leaf_max_size; ++i) {
      const auto dist =
          KernelFunc(leaf_node_table[leaf_id * leaf_max_size + i], q_point);

      my_min = std::min(my_min, dist);
    }

    out[q_idx] = std::min(out[q_idx], my_min);
  }
}

}  // namespace redwood::accelerator