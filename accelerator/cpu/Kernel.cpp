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
    if (leaf_id < 0) continue;  // slot not active this pass (uid-indexed batch)
    const T q = u_q[i];

    auto my_min = std::numeric_limits<float>::max();
    for (int j = 0; j < max_leaf_size; ++j) {
      const float dist = functor(u_lnt[leaf_id * max_leaf_size + j], q);
      my_min = std::min(my_min, dist);
    }

    u_out[i] = std::min(u_out[i], my_min);
  }
}

// Serial reference for the Barnes-Hut leaf reduction: sum of interactions over
// the leaf's points, folded into the running result.
template <typename T, typename Functor>
void BarnesKernel(int /*tid*/, int /*stream_id*/, const T* u_lnt,
                  int max_leaf_size, const T* u_q, const int* u_node_idx,
                  int num_active, float* u_out, Functor functor) {
  for (int i = 0; i < num_active; ++i) {
    const int leaf_id = u_node_idx[i];
    const T q = u_q[i];

    float my_sum = 0.0f;
    for (int j = 0; j < max_leaf_size; ++j) {
      my_sum += functor(q, u_lnt[leaf_id * max_leaf_size + j]);
    }
    u_out[i] += my_sum;
  }
}

// Insert a candidate distance into a sorted-ascending top-K set in place.
// Shared shape with the CUDA/SYCL device versions (no std::lower_bound, so it
// is trivially portable to device code).
template <int K>
inline void InsertTopK(float* rank, float dist) {
  if (dist >= rank[K - 1]) return;
  int pos = K - 1;
  while (pos > 0 && rank[pos - 1] > dist) {
    rank[pos] = rank[pos - 1];
    --pos;
  }
  rank[pos] = dist;
}

// Serial reference for the KNN leaf reduction: merge the leaf's distances into
// the running sorted K-nearest set at u_out + i*K.
template <typename T, int K, typename Functor>
void KnnKernel(int /*tid*/, int /*stream_id*/, const T* u_lnt,
               int max_leaf_size, const T* u_q, const int* u_node_idx,
               int num_active, float* u_out, Functor functor) {
  for (int i = 0; i < num_active; ++i) {
    const int leaf_id = u_node_idx[i];
    const T q = u_q[i];
    float* rank = u_out + i * K;

    for (int j = 0; j < max_leaf_size; ++j) {
      const float dist = functor(u_lnt[leaf_id * max_leaf_size + j], q);
      InsertTopK<K>(rank, dist);
    }
  }
}

// Instantiate the combinations the applications actually use, matching the
// CUDA/SYCL backends' explicit instantiation lists.
template void NearestNeighborKernel<Point4F, dist::Euclidean>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Euclidean functor);

template void BarnesKernel<Point4F, dist::Gravity>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Gravity functor);

template void KnnKernel<Point4F, 32, dist::Euclidean>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Euclidean functor);

}  // namespace redwood
