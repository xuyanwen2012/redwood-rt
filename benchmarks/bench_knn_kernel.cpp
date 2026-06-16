// Benchmark: KNN leaf reduction (redwood::KnnKernel<Point4F,32>) across the
// linked backend. Wall-clock per iteration = launch + execute + sync.
#include <benchmark/benchmark.h>

#include <limits>

#include "Redwood/Kernel.hpp"
#include "bench_common.hpp"

static constexpr int kK = 32;  // must match the backends' KnnKernel instantiation

static void BM_KnnKernel(benchmark::State& state) {
  const int leaf = static_cast<int>(state.range(0));
  const int active = static_cast<int>(state.range(1));
  constexpr float kInf = std::numeric_limits<float>::max();
  constexpr dist::Euclidean functor;

  bench::EnsureInit();
  bench::KernelInputs in;
  in.Alloc(/*leaves=*/1024, leaf, active, /*k=*/kK, /*bodies=*/false);

  // Warm up.
  in.ResetOut(kInf);
  redwood::KnnKernel<Point4F, kK>(0, 0, in.u_lnt, leaf, in.u_q, in.u_node_idx,
                                  active, in.u_out, functor);
  redwood::DeviceSynchronize();

  for (auto _ : state) {
    state.PauseTiming();
    // Reset so each iteration merges into a fresh (+inf) K-set — otherwise the
    // insert branch short-circuits after the first iteration.
    in.ResetOut(kInf);
    state.ResumeTiming();

    redwood::KnnKernel<Point4F, kK>(0, 0, in.u_lnt, leaf, in.u_q, in.u_node_idx,
                                    active, in.u_out, functor);
    redwood::DeviceSynchronize();
    benchmark::DoNotOptimize(in.u_out[0]);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * active *
                          leaf);
  in.Free();
}

BENCHMARK(BM_KnnKernel)->REDWOOD_BENCH_ARGS;
