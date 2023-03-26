#pragma once

#include "Point.hpp"

namespace redwood {

enum class DistanceMetrics {
  Euclidean,
  Manhattan,
  Chebyshev,
  Gravity,
  Gaussian,
  Tophat
};

void LaunchNnKenrnel(const int* u_leaf_indices, /**/
                     const Point4F* u_q_points, /**/
                     int num_active_leafs,      /**/
                     float* u_out,              /* stream base addr */
                     const Point4F* u_lnt_data, /**/
                     int max_leaf_size, int stream_id);

template <typename T, typename Functor>
void NearestNeighborKernel(int stream_id, const T* lnt, int max_leaf_size,
                           const T* u_q, const int* u_node_idx, int num_active,
                           float* u_out, Functor functor);

}  // namespace redwood