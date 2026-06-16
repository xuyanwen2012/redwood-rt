#include <iostream>
#include <vector>

#include "CudaUtils.cuh"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"
#include "nn/Reductions.cuh"

namespace redwood {

extern std::vector<cudaStream_t> streams;

////////////////////////////////////////////////////////////////////////////////
// Device kernels for the Barnes-Hut and KNN leaf reductions
//
// One thread per active slot. These mirror the CPU/SYCL backends' semantics so
// the same brute-force-oracle tests validate all three. (The NN path keeps its
// existing warp-reduction kernel, FindMinDistWarp6.)
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Functor>
__global__ void BarnesKernelImpl(const T* lnt, const T* q, const int* node_idx,
                                 float* out, int num_active, int max_leaf_size,
                                 Functor functor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_active) return;
  const int leaf_id = node_idx[i];
  const T qi = q[i];

  float my_sum = 0.0f;
  for (int j = 0; j < max_leaf_size; ++j) {
    my_sum += functor(qi, lnt[leaf_id * max_leaf_size + j]);
  }
  out[i] += my_sum;
}

template <typename T, int K, typename Functor>
__global__ void KnnKernelImpl(const T* lnt, const T* q, const int* node_idx,
                              float* out, int num_active, int max_leaf_size,
                              Functor functor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_active) return;
  const int leaf_id = node_idx[i];
  const T qi = q[i];

  // Load the running K-nearest set into registers, merge, write back.
  float rank[K];
#pragma unroll
  for (int t = 0; t < K; ++t) rank[t] = out[i * K + t];

  for (int j = 0; j < max_leaf_size; ++j) {
    const float dist = functor(lnt[leaf_id * max_leaf_size + j], qi);
    if (dist < rank[K - 1]) {
      int pos = K - 1;
      while (pos > 0 && rank[pos - 1] > dist) {
        rank[pos] = rank[pos - 1];
        --pos;
      }
      rank[pos] = dist;
    }
  }

#pragma unroll
  for (int t = 0; t < K; ++t) out[i * K + t] = rank[t];
}

// Nearest-neighbor leaf reduction, grid-stride one-thread-per-slot (mirrors the
// BH/KNN kernels). Replaces the single-block warp kernel FindMinDistWarp6, which
// launched dim_grid(1,1,1) and so used only one SM regardless of GPU size --
// fine for the original fixed 1024-wide batch, but it cannot scale with
// num_active. This version covers the whole GPU.
template <typename T, typename Functor>
__global__ void NnMinKernelImpl(const T* lnt, const T* q, const int* node_idx,
                                float* out, int num_active, int max_leaf_size,
                                Functor functor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_active) return;
  const int leaf_id = node_idx[i];
  if (leaf_id < 0) return;  // slot not active this pass (uid-indexed batch)
  const T qi = q[i];

  float my_min = std::numeric_limits<float>::max();
  for (int j = 0; j < max_leaf_size; ++j) {
    const float dist = functor(lnt[leaf_id * max_leaf_size + j], qi);
    my_min = fminf(my_min, dist);
  }
  out[i] = fminf(out[i], my_min);
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper functions for kernel launch
////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Functor>
void NearestNeighborKernel(const int tid, int stream_id, const T* u_lnt,
                           int max_leaf_size, const T* u_q,
                           const int* u_node_idx, int num_active, float* u_out,
                           Functor functor) {
  constexpr int kBlock = 256;
  const int grid = (num_active + kBlock - 1) / kBlock;
  const auto my_stream_id = tid * kNumStreams + stream_id;
  NnMinKernelImpl<<<grid, kBlock, 0, streams[my_stream_id]>>>(
      u_lnt, u_q, u_node_idx, u_out, num_active, max_leaf_size, functor);
}

template <typename T, typename Functor>
void BarnesKernel(const int tid, int stream_id, const T* u_lnt,
                  int max_leaf_size, const T* u_q, const int* u_node_idx,
                  int num_active, float* u_out, Functor functor) {
  constexpr int kBlock = 256;
  const int grid = (num_active + kBlock - 1) / kBlock;
  const auto my_stream_id = tid * kNumStreams + stream_id;
  BarnesKernelImpl<<<grid, kBlock, 0, streams[my_stream_id]>>>(
      u_lnt, u_q, u_node_idx, u_out, num_active, max_leaf_size, functor);
}

template <typename T, int K, typename Functor>
void KnnKernel(const int tid, int stream_id, const T* u_lnt, int max_leaf_size,
               const T* u_q, const int* u_node_idx, int num_active,
               float* u_out, Functor functor) {
  constexpr int kBlock = 256;
  const int grid = (num_active + kBlock - 1) / kBlock;
  const auto my_stream_id = tid * kNumStreams + stream_id;
  KnnKernelImpl<T, K><<<grid, kBlock, 0, streams[my_stream_id]>>>(
      u_lnt, u_q, u_node_idx, u_out, num_active, max_leaf_size, functor);
}

// Instantiating the ones we are using
template void NearestNeighborKernel<Point4F, dist::Euclidean>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Euclidean functor_type);

template void BarnesKernel<Point4F, dist::Gravity>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Gravity functor);

template void KnnKernel<Point4F, 32, dist::Euclidean>(
    int tid, int stream_id, const Point4F* u_lnt, int max_leaf_size,
    const Point4F* u_q, const int* u_node_idx, int num_active, float* u_out,
    dist::Euclidean functor);

}  // namespace redwood