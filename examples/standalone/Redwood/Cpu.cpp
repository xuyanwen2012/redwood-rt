#include <iostream>

#include "../DistanceMetrics.hpp"
#include "Kernels.hpp"
#include "Redwood.hpp"

namespace redwood {

void Init() {
  // No Op
}

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

void LaunchNnKenrnel(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* u_out,               /* stream base addr */
                     const Point4F* u_lnt_data,  /**/
                     const int max_leaf_size, const int stream_id) {
  constexpr dist::Euclidean functor;

  // i is batch id, = tid, = index in the buffer
  for (int i = 0; i < num_active_leafs; ++i) {
    const auto node_idx = u_leaf_indices[i];
    const auto q = u_q_points[i];

    const auto node_addr = u_lnt_data + node_idx * max_leaf_size;

    for (int j = 0; j < max_leaf_size; ++j) {
      const float dist = functor(node_addr[j], q);

      u_out[i] = std::min(u_out[i], dist);
    }
  }
}

}  // namespace redwood