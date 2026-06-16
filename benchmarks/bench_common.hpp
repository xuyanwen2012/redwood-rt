#pragma once
// Shared setup for the kernel benchmarks. Mirrors the test fixtures
// (tests/test_*_kernel/reduction.cpp) but sized for performance measurement and
// reused across the three algorithm benchmarks.
#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>

#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"

namespace bench {

// Initialize the backend exactly once per process (creates CUDA streams / SYCL
// queues). Safe to call from every benchmark function.
inline void EnsureInit() {
  static const bool once = [] {
    redwood::Init(/*num_threads=*/1);
    return true;
  }();
  (void)once;
}

inline Point4F RandPoint(std::mt19937& rng) {
  std::uniform_real_distribution<float> d(0.f, 1024.f);
  return Point4F{{d(rng), d(rng), d(rng), d(rng)}};
}

inline Point4F RandBody(std::mt19937& rng) {
  std::uniform_real_distribution<float> pos(0.f, 1024.f);
  std::uniform_real_distribution<float> mass(0.1f, 10.f);
  return Point4F{{pos(rng), pos(rng), pos(rng), mass(rng)}};
}

// Owns the USM buffers the leaf-reduction kernels consume:
//   u_lnt      : num_leaves * max_leaf_size points (the leaf-node table)
//   u_q        : num_active query points
//   u_node_idx : num_active leaf assignments (which leaf each slot reduces)
//   u_out      : num_active * k result floats (k=1 for NN/BH, K for KNN)
// Allocated once per benchmark invocation (outside the timed loop) and filled
// with deterministic seeded data so runs are comparable.
struct KernelInputs {
  int num_leaves = 0;
  int max_leaf_size = 0;
  int num_active = 0;
  int k = 1;

  Point4F* u_lnt = nullptr;
  Point4F* u_q = nullptr;
  int* u_node_idx = nullptr;
  float* u_out = nullptr;

  void Alloc(int leaves, int leaf, int active, int k_, bool bodies) {
    num_leaves = leaves;
    max_leaf_size = leaf;
    num_active = active;
    k = k_;

    std::mt19937 rng(12345);
    const std::size_t lnt_n =
        static_cast<std::size_t>(num_leaves) * max_leaf_size;
    u_lnt = redwood::UsmMalloc<Point4F>(lnt_n);
    for (std::size_t i = 0; i < lnt_n; ++i)
      u_lnt[i] = bodies ? RandBody(rng) : RandPoint(rng);

    u_q = redwood::UsmMalloc<Point4F>(num_active);
    u_node_idx = redwood::UsmMalloc<int>(num_active);
    u_out = redwood::UsmMalloc<float>(static_cast<std::size_t>(num_active) * k);

    std::uniform_int_distribution<int> pick(0, num_leaves - 1);
    for (int i = 0; i < num_active; ++i) {
      u_q[i] = bodies ? RandBody(rng) : RandPoint(rng);
      u_node_idx[i] = pick(rng);
    }
  }

  // Reset the folded output buffer to the kernel's identity element so every
  // timed iteration does the same work (NN/KNN: +inf, BH: 0).
  void ResetOut(float identity) {
    const std::size_t n = static_cast<std::size_t>(num_active) * k;
    for (std::size_t i = 0; i < n; ++i) u_out[i] = identity;
  }

  void Free() {
    redwood::UsmFree(u_lnt);
    redwood::UsmFree(u_q);
    redwood::UsmFree(u_node_idx);
    redwood::UsmFree(u_out);
  }
};

// Args applied to every algorithm benchmark. Each point is (leaf, num_active):
//   * leaf-size study  : sweep leaf {32..1024} at num_active=1024 (paper Table III knob)
//   * scaling study    : sweep num_active {16K, 256K} at leaf=128
// The scaling study is what saturates a GPU -- at num_active=1024 only a handful
// of thread blocks launch, so the GPU sits ~99% idle and the comparison is
// dominated by launch overhead. CPU kernels here are single-threaded serial.
inline void RedwoodArgs(benchmark::internal::Benchmark* b) {
  b->Unit(benchmark::kMicrosecond)->ArgNames({"leaf", "active"});
  for (int leaf : {32, 64, 128, 256, 512, 1024}) b->Args({leaf, 1024});
  for (int active : {16384, 262144}) b->Args({128, active});
}

}  // namespace bench
