#include <iostream>

#include "../Kernel.hpp"
#include "../Point.hpp"
#include "CudaUtils.cuh"
#include "nn/Reductions.cuh"

namespace redwood {

extern cudaStream_t streams[kNumStreams];

void LaunchNnKenrnel(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* u_out,               /* stream base addr */
                     const Point4F* u_lnt_data,  /**/
                     const int max_leaf_size, const int stream_id) {
  const auto n_blocks = 1;
  constexpr auto n_threads = 1024;
  constexpr auto smem_size = 0;

  const dist_cuda::Euclidean functor;

  FindMinDistWarp6<<<n_blocks, n_threads, smem_size, streams[stream_id]>>>(
      u_lnt_data, u_q_points, u_leaf_indices, u_out, num_active_leafs,
      max_leaf_size, functor);
}

}  // namespace redwood