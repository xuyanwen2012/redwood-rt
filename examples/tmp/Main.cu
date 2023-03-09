
#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "../Utils.hpp"
#include "CuUtils.hpp"
#include "cuda_runtime.h"

namespace cg = cooperative_groups;

constexpr auto kNumBlocks = 1;

inline __host__ __device__ float EuclideanDistance(const float2 p,
                                                   const float2 q) {
  const auto dx = p.x - q.x;
  const auto dy = p.y - q.y;
  return sqrtf(dx * dx + dy * dy);
}

inline __host__ __device__ float HaversineDistance(const float2 p,
                                                   const float2 q) {
  float lat1 = p.x;
  float lon1 = p.y;
  float lat2 = q.x;
  float lon2 = q.y;

  float dLat = (lat2 - lat1) * M_PI / 180.0f;
  float dLon = (lon2 - lon1) * M_PI / 180.0f;

  lat1 = lat1 * M_PI / 180.0f;
  lat2 = lat2 * M_PI / 180.0f;

  const float a = powf(sinf(dLat / 2), 2) +
                  powf(sinf(dLon / 2), 2) * cosf(lat1) * cosf(lat2);
  const float c = 2 * asinf(sqrtf(a));

  const float radius = 6371;  // Earth's radius in kilometers

  return radius * c;
}

__device__ __forceinline__ void WaitCPU(int* com) {
  int block_id = blockIdx.x;
  while (com[block_id] != 1 && com[kNumBlocks] != 1) {
    __threadfence();
  }
}

__device__ __forceinline__ void WorkComplete(int* com) {
  int block_id = blockIdx.x;
  com[block_id] = 0;
}

//--expt-relaxed-constexpr
// auto my_min = std::numeric_limits<float>::max();
template <typename DataT, typename ResultT>
__device__ void FunctionKernel(cg::thread_group g, DataT* u_buffer, const int n,
                               ResultT* u_result, const DataT q) {
  constexpr int block_threads = 1024;
  using BlockReduce = cub::BlockReduce<float, block_threads>;

  // This need to be a conexpr, because I am passing this as a template argument
  // for Cub library. Although is is just the same as 'g.size()'
  __shared__ union { BlockReduce::TempStorage reduce; } temp_storage;

  const auto tid = g.thread_rank();
  const auto size = g.size();  // block_threads

  float thread_value[1];
  thread_value[0] = 999999999999.9f;

  int global_id = tid;
  for (; global_id < n; global_id += size) {
    const auto p = u_buffer[global_id];
    const auto dist = EuclideanDistance(p, q);
    thread_value[0] = min(thread_value[0], dist);
  }

  float aggregate =
      BlockReduce(temp_storage.reduce).Reduce(thread_value, cub::Min());

  if (tid == 0) u_result[0] = min(u_result[0], aggregate);
}

template <typename DataT, typename ResultT>
__global__ void PersistentKernel(DataT* u_buffer, const int n, const DataT q,
                                 ResultT* u_result, int* com) {
  auto cta = cg::this_thread_block();
  const auto tid = cta.thread_rank();

  while (com[kNumBlocks] != 1) {
    if (tid == 0) WaitCPU(com);
    __syncthreads();

    // cancelling point
    if (com[kNumBlocks] == 1) return;

    FunctionKernel(cta, u_buffer, n, u_result, q);

    if (tid == 0) WorkComplete(com);
  }
}

void StartGPU(int* com) {
  // printf("StartGPU\n");
  com[0] = 1;
}

void WaitGPU(int* com) {
  // printf("WaitGPU\n");
  int sum;
  do {
    sum = 0;
    asm volatile("" ::: "memory");
    sum |= com[0];
  } while (sum != 0);
}

void EndGPU(int* com) {
  printf("cpu is ending GPU\n");
  com[kNumBlocks] = 1;
}

static float2 RandomPoint() {
  static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
  static std::uniform_real_distribution<float> lat_dist(-90.0f, 90.0f);
  static std::uniform_real_distribution<float> lon_dist(-180.0f, 180.0f);

  return {lat_dist(generator), lon_dist(generator)};
}

std::vector<float2> GenerateRandomPoints(const int num_points) {
  std::vector<float2> points(num_points);
  std::generate(points.begin(), points.end(), RandomPoint);
  return points;
}

float2* tmp = nullptr;

int main() {
  constexpr int n = 1024 * 1024;
  // constexpr int m = 1024 * 1024;

  constexpr int buffer_size = 1024;

  float2* u_buffer = nullptr;
  float* u_result = nullptr;
  int* u_com = nullptr;

  cudaAllocMapped(&u_buffer, sizeof(float2) * buffer_size);
  cudaAllocMapped(&u_result, sizeof(float) * 1);
  cudaAllocMapped(&u_com, sizeof(int) * (kNumBlocks + 1));

  u_result[0] = std::numeric_limits<float>::max();

  auto h_p_data = GenerateRandomPoints(n);
  const float2 q{0.0f, 0.0f};

  TimeTask("CPU Compute: ", [&] {
    auto sum = std::numeric_limits<float>::max();

    for (int i = 0; i < n; ++i) {
      const auto dist = EuclideanDistance(h_p_data[i], q);
      // if (i <= 32) std::cout << dist << std::endl;
      sum = std::min(sum, dist);
    }

    std::cout << "Ground truth: " << sum << std::endl;
  });

  tmp = (float2*)malloc(sizeof(float2) * n);
  TimeTask("CPU Memcpy: ",
           [&] { memcpy(tmp, h_p_data.data(), sizeof(float2) * n); });

  constexpr auto num_threads = 1024;
  PersistentKernel<float2, float>
      <<<kNumBlocks, num_threads>>>(u_buffer, num_threads, q, u_result, u_com);

  TimeTask("PK GPU: ", [&] {
    const auto iterations = n / num_threads;
    for (int i = 0; i < iterations; ++i) {
      // std::cout << "\nIteration: (" << i << '/' << iterations << ')' <<
      // std::endl;
      memcpy(u_buffer, h_p_data.data() + i * num_threads,
             sizeof(float2) * num_threads);

      StartGPU(u_com);
      WaitGPU(u_com);
    }
  });

  std::cout << "\tMin sofar: " << u_result[0] << std::endl;

  //   std::iota(u_buffer, u_buffer + n, i * n);

  //   for (int i = 0; i < 8; ++i) {
  //     std::cout << i << ": " << u_buffer[i] << std::endl;
  //   }
  //   std::cout << "..." << std::endl;
  //   for (int i = n - 8; i < n; ++i) {
  //     std::cout << i << ": " << u_buffer[i] << std::endl;
  //   }
  // }

  // // std::chrono::seconds dura(2);
  // // std::this_thread::sleep_for(dura);

  EndGPU(u_com);

  HANDLE_ERROR(cudaFreeHost(u_buffer));
  HANDLE_ERROR(cudaFreeHost(u_com));

  return 0;
}