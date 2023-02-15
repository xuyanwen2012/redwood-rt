#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cub/cub.cuh>
#include <limits>

#include "../CudaUtils.cuh"
#include "cuda_runtime.h"

namespace cg = cooperative_groups;

template <int kK>
__global__ void DebugSort(const float* input, const int n, float* out) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  // Need to load this from out first.
  float my_mins[kK];
  for (int i = 0; i < kK; ++i) {
    my_mins[i] = out[i];
    printf("my_mins[%d] %f\n", i, my_mins[i]);
  }

  for (int i = 0; i < n; ++i) {
    const auto dist = input[i];

    // find min 'idx' first
    int i = 0;
    while (i < kK && dist >= my_mins[i]) {
      i++;
    }

    if (i < kK) {
      // Shift the others back
      for (int j = kK - 1; j > i; j--) {
        my_mins[j] = my_mins[j - 1];
      }
      my_mins[i] = dist;
    }
  }

  // Write back
  for (int i = 0; i < kK; ++i) {
    out[i] = my_mins[i];
  }
}

int main() {
  constexpr auto n = 64;
  constexpr auto k = 32;

  float* u_input;
  float* u_out;
  HANDLE_ERROR(cudaMallocManaged(&u_input, n * sizeof(float)));
  HANDLE_ERROR(cudaMallocManaged(&u_out, k * sizeof(float)));

  std::fill_n(u_out, k, std::numeric_limits<float>::max());

  for (int i = 0; i < n; ++i) u_input[i] = i + 0.5f;
  for (int i = 0; i < 6; ++i) {
    u_out[i] = i;
  }

  constexpr auto n_blocks = 1;
  constexpr auto n_threads = 1;
  constexpr auto smem_size = 0;
  DebugSort<k><<<n_blocks, n_threads, smem_size>>>(u_input, n, u_out);
  HANDLE_ERROR(cudaDeviceSynchronize());

  for (int i = 0; i < k; ++i) {
    std::cout << i << ": " << u_out[i] << std::endl;
  }

  cudaFree(u_input);
  cudaFree(u_out);

  return 0;
}