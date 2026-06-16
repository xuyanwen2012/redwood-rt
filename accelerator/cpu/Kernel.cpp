#include "Redwood/Kernel.hpp"

#include <algorithm>
#include <limits>

#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Point.hpp"

namespace redwood {

// Serial reference implementation of the nearest-neighbor leaf-node reduction.
//
// This mirrors the externally-observable behaviour of the CUDA kernel
// (CudaNnNaive / FindMinDistWarp6): for each active query slot, compute the
// minimum distance to the points in its assigned leaf node, then fold that into
// the running result with a min(). Because it runs synchronously, the matching
// DeviceStreamSynchronize() in Core.cpp is a no-op.
template <typename T, typename Functor>
void NearestNeighborKernel(int /*tid*/, int /*stream_id*/, const T* u_lnt,
                           int max_leaf_size, const T* u_q,
                           const int* u_node_idx, int num_active, float* u_out,
                           Functor functor) {
  for (int i = 0; i < num_active; ++i) {
    const int leaf_id = u_node_idx[i];
    const T q = u_q[i];

    auto my_min = std::numeric_limits<float>::max();
    for (int j = 0; j < max_leaf_size; ++j) {
      const float dist = functor(u_lnt[leaf_id * max_leaf_size + j], q);
      my_min = std::min(my_min, dist);
    }

    u_out[i] = std::min(u_out[i], my_min);
  }
}

// Instantiate the combinations the applications actually use, matching the
// CUDA backend's explicit instantiation list.
template void NearestNeighborKernel<Point4F, dist::Euclidean>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Euclidean functor);

}  // namespace redwood
