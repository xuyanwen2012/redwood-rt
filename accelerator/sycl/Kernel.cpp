#include "Redwood/Kernel.hpp"

#include <sycl/sycl.hpp>

#include <limits>

#include "Consts.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Point.hpp"

extern sycl::queue g_queues[kNumStreams];

namespace redwood {

// SYCL implementation of the nearest-neighbor leaf-node reduction, matching the
// unified redwood:: API (and the CPU/CUDA backends' semantics): one work-item
// per active query slot computes the min distance to its leaf's points and
// folds it into the running result. Submitted async to the stream's queue;
// the matching DeviceStreamSynchronize() waits on it.
template <typename T, typename Functor>
void NearestNeighborKernel(int /*tid*/, int stream_id, const T* u_lnt,
                           int max_leaf_size, const T* u_q,
                           const int* u_node_idx, int num_active, float* u_out,
                           Functor functor) {
  if (num_active <= 0) return;
  g_queues[stream_id].submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(num_active), [=](sycl::id<1> idx) {
      const int i = static_cast<int>(idx[0]);
      const int leaf_id = u_node_idx[i];
      if (leaf_id < 0) return;  // slot not active this pass (uid-indexed batch)
      const T q = u_q[i];

      float my_min = std::numeric_limits<float>::max();
      for (int j = 0; j < max_leaf_size; ++j) {
        const float dist = functor(u_lnt[leaf_id * max_leaf_size + j], q);
        my_min = sycl::min(my_min, dist);
      }
      u_out[i] = sycl::min(u_out[i], my_min);
    });
  });
}

// Barnes-Hut leaf reduction: one work-item per active slot sums the leaf's
// interactions and folds the result into u_out[i].
template <typename T, typename Functor>
void BarnesKernel(int /*tid*/, int stream_id, const T* u_lnt, int max_leaf_size,
                  const T* u_q, const int* u_node_idx, int num_active,
                  float* u_out, Functor functor) {
  if (num_active <= 0) return;
  g_queues[stream_id].submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(num_active), [=](sycl::id<1> idx) {
      const int i = static_cast<int>(idx[0]);
      const int leaf_id = u_node_idx[i];
      const T q = u_q[i];

      float my_sum = 0.0f;
      for (int j = 0; j < max_leaf_size; ++j) {
        my_sum += functor(q, u_lnt[leaf_id * max_leaf_size + j]);
      }
      u_out[i] += my_sum;
    });
  });
}

// KNN leaf reduction: one work-item per active slot merges the leaf's distances
// into the running sorted K-nearest set at u_out + i*K.
template <typename T, int K, typename Functor>
void KnnKernel(int /*tid*/, int stream_id, const T* u_lnt, int max_leaf_size,
               const T* u_q, const int* u_node_idx, int num_active,
               float* u_out, Functor functor) {
  if (num_active <= 0) return;
  g_queues[stream_id].submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(num_active), [=](sycl::id<1> idx) {
      const int i = static_cast<int>(idx[0]);
      const int leaf_id = u_node_idx[i];
      const T q = u_q[i];
      float* rank = u_out + i * K;

      for (int j = 0; j < max_leaf_size; ++j) {
        const float dist = functor(u_lnt[leaf_id * max_leaf_size + j], q);
        if (dist < rank[K - 1]) {
          int pos = K - 1;
          while (pos > 0 && rank[pos - 1] > dist) {
            rank[pos] = rank[pos - 1];
            --pos;
          }
          rank[pos] = dist;
        }
      }
    });
  });
}

// Match the explicit instantiation list of the other backends.
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
