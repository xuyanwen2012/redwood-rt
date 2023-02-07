#include <CL/sycl.hpp>

#include "../Kernel.hpp"

constexpr auto kNumStreams = 2;
extern sycl::device device;
extern sycl::context ctx;
extern sycl::queue qs[kNumStreams];
extern const Point4F* usm_leaf_node_table;

inline float kernel_func(const Point4F p, const Point4F q) {
  auto dist = float();

  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

namespace redwood::internal {

// Main entry to the NN Kernel
void ProcessNnBuffer(const void* query_points, const int* query_idx,
                     const int* leaf_idx, float* out, const int num,
                     const int leaf_max_size, const int stream_id) {
  auto my_query_points = static_cast<const Point4F*>(query_points);

  const auto my_leaf_node_table = usm_leaf_node_table;

  qs[stream_id].submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range(num), [=](const sycl::id<1> idx) {
      const auto leaf_id = leaf_idx[idx];
      const auto q_point = my_query_points[idx];
      const auto q_idx = query_idx[idx];

      auto my_min = std::numeric_limits<float>::max();
      for (int i = 0; i < leaf_max_size; ++i) {
        const auto dist = kernel_func(
            my_leaf_node_table[leaf_id * leaf_max_size + i], q_point);

        my_min = sycl::min(my_min, dist);
      }

      out[q_idx] = sycl::min(out[q_idx], my_min);
    });
  });
}

}  // namespace redwood::internal