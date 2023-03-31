#include <iostream>

#include "CudaUtils.cuh"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"
#include "nn/Reductions.cuh"

namespace redwood {

extern cudaStream_t streams[kNumStreams];

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Functor>
void NearestNeighborKernel(int stream_id, const T* u_lnt, int max_leaf_size,
                           const T* u_q, const int* u_node_idx, int num_active,
                           float* u_out, Functor functor) {
  constexpr dim3 dim_grid(1, 1, 1);
  constexpr dim3 dim_block(1024, 1, 1);
  constexpr auto smem_size = 0;
  FindMinDistWarp6<<<dim_grid, dim_block, smem_size, streams[stream_id]>>>(
      u_lnt, u_q, u_node_idx, u_out, num_active, max_leaf_size, functor);
}

// Instantiating the ones we are using
template void NearestNeighborKernel<Point4F, dist::Euclidean>(
    int stream_id, const Point4F* u_lnt, int max_leaf_size, const Point4F* u_q,
    const int* u_node_idx, int num_active, float* u_out,
    dist::Euclidean functor_type);

}  // namespace redwood