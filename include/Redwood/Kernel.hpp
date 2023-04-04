#pragma once

#include "Point.hpp"

namespace redwood {

template <typename T, typename Functor>
void NearestNeighborKernel(int tid, int stream_id, const T* u_lnt,
                           int max_leaf_size, const T* u_q,
                           const int* u_node_idx, int num_active, float* u_out,
                           Functor functor);

}  // namespace redwood