#pragma once

#include "Point.hpp"

namespace redwood {

template <typename T, typename Functor>
void NearestNeighborKernel(int tid, int stream_id, const T* u_lnt,
                           int max_leaf_size, const T* u_q,
                           const int* u_node_idx, int num_active, float* u_out,
                           Functor functor);

// Barnes-Hut leaf reduction. For each active slot i, accumulate the sum of
// interactions functor(u_q[i], point) over the points of leaf u_node_idx[i]
// and fold it into u_out[i] (u_out[i] += sum). Matches the per-leaf
// accumulation in examples/barnes/ReducerHandler.hpp.
template <typename T, typename Functor>
void BarnesKernel(int tid, int stream_id, const T* u_lnt, int max_leaf_size,
                  const T* u_q, const int* u_node_idx, int num_active,
                  float* u_out, Functor functor);

// KNN leaf reduction. For each active slot i, merge the distances from u_q[i]
// to the points of leaf u_node_idx[i] into the running K-nearest set stored at
// u_out + i*K (sorted ascending; u_out[i*K + K-1] is the current k-th nearest).
// K is a compile-time constant (the paper fixes k = 32).
template <typename T, int K, typename Functor>
void KnnKernel(int tid, int stream_id, const T* u_lnt, int max_leaf_size,
               const T* u_q, const int* u_node_idx, int num_active,
               float* u_out, Functor functor);

}  // namespace redwood