#include <cmath>
#include <iostream>
#include <limits>

#include "Redwood/Point.hpp"

namespace redwood {

void Init() {}

void DeviceStreamSynchronize(const int stream_id) {
  // No Op
}

void DeviceSynchronize() {
  // No Op
}

void AttachStreamMem(const int stream_id, void* addr) {
  // No Op
}

void* UsmMalloc(std::size_t n) {
  std::cout << "std::malloc() " << n << std::endl;
  return malloc(n);
}

void UsmFree(void* ptr) {
  std::cout << "std::free() " << ptr << std::endl;
  if (ptr) {
    free(ptr);
  }
}

inline auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

// Modified version
inline float KernelFuncBh(const Point4F p, const Point4F q) {
  const auto dx = p.data[0] - q.data[0];
  const auto dy = p.data[1] - q.data[1];
  const auto dz = p.data[2] - q.data[2];
  const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
  const auto inv_dist = rsqrtf(dist_sqr);
  const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
  const auto with_mass = inv_dist3 * p.data[3];
  return dx * with_mass + dy * with_mass + dz * with_mass;
}

inline float KernelFuncKnn(const Point4F p, const Point4F q) {
  auto dist = float();

  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

// Naive
inline float SumLeaf(const Point4F q_point, const Point4F* base,
                     const int leaf_max_size) {
  float acc{};
  for (int i = 0; i < leaf_max_size; ++i) acc += KernelFuncBh(base[i], q_point);
  return acc;
}

void ComputeOneBatchAsync(const int* u_leaf_indices,  /**/
                          const int num_active_leafs, /**/
                          float* out,                 /**/
                          const Point4F* u_lnt_data,  /**/
                          const int* u_lnt_sizes,     /**/
                          const Point4F q,            /**/
                          const int stream_id) {
  constexpr auto leaf_max_size = 64;
  for (int i = 0; i < num_active_leafs; ++i) {
    const auto leaf_id = u_leaf_indices[i];

    auto acc = SumLeaf(q, u_lnt_data + leaf_id * leaf_max_size, leaf_max_size);

    *out += acc;
  }
}
void ComputeOneBatchAsync_PB(const int* u_leaf_indices,  /**/
                          const int num_active_leafs, /**/
                          float* out,                 /**/
                          const Point4F* u_lnt_data,  /**/
                          const int* u_lnt_sizes,     /**/
                          const Point4F q,            /**/
                          const int stream_id,
                          const int pb_idx) {
  constexpr auto leaf_max_size = 64;
  for (int i = 0; i < num_active_leafs; ++i) {
    const auto leaf_id = u_leaf_indices[i];

    auto acc = SumLeaf(q, u_lnt_data + leaf_id * leaf_max_size, leaf_max_size);

    out[pb_idx] += acc;
  }
}

void ProcessKnnAsync(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* out,                 /**/
                     const Point4F* u_lnt_data,  /**/
                     const int* u_lnt_sizes,     /**/
                     const int stream_id) {}

}  // namespace redwood