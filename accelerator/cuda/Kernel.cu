#include <iostream>

#include "CudaUtils.cuh"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"
#include "nn/Reductions.cuh"

namespace redwood {

extern cudaStream_t streams[kNumStreams];

void LaunchNnKenrnel(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* u_out,               /* stream base addr */
                     const Point4F* u_lnt_data,  /**/
                     const int max_leaf_size, const int stream_id) {
  const dist_cuda::Euclidean functor;

  constexpr dim3 dim_grid(1, 1, 1);
  constexpr dim3 dim_block(1024, 1, 1);
  constexpr auto smem_size = 0;
  FindMinDistWarp6<<<dim_grid, dim_block, smem_size, streams[stream_id]>>>(
      u_lnt_data, u_q_points, u_leaf_indices, u_out, num_active_leafs,
      max_leaf_size, functor);
}

}  // namespace redwood