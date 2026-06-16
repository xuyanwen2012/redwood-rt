// Benchmark: nearest-neighbor leaf reduction (redwood::NearestNeighborKernel)
// across the linked backend. Wall-clock per iteration = launch + execute + sync.
#include <benchmark/benchmark.h>

#include <limits>

#include "Redwood/Kernel.hpp"
#include "bench_common.hpp"

static void BM_NnKernel(benchmark::State& state) {
  const int leaf = static_cast<int>(state.range(0));
  const int active = static_cast<int>(state.range(1));
  constexpr float kInf = std::numeric_limits<float>::max();
  constexpr dist::Euclidean functor;

  bench::EnsureInit();
  bench::KernelInputs in;
  in.Alloc(/*leaves=*/1024, leaf, active, /*k=*/1, /*bodies=*/false);

  // Warm up (CUDA first-launch/JIT, SYCL kernel build).
  in.ResetOut(kInf);
  redwood::NearestNeighborKernel(0, 0, in.u_lnt, leaf, in.u_q, in.u_node_idx,
                                 active, in.u_out, functor);
  redwood::DeviceSynchronize();

  for (auto _ : state) {
    state.PauseTiming();
    in.ResetOut(kInf);
    state.ResumeTiming();

    redwood::NearestNeighborKernel(0, 0, in.u_lnt, leaf, in.u_q, in.u_node_idx,
                                   active, in.u_out, functor);
    redwood::DeviceSynchronize();
    benchmark::DoNotOptimize(in.u_out[0]);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * active *
                          leaf);
  in.Free();
}

BENCHMARK(BM_NnKernel)->REDWOOD_BENCH_ARGS;
