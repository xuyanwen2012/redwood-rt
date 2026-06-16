// Benchmark: Barnes-Hut leaf reduction (redwood::BarnesKernel) across the linked
// backend. Wall-clock per iteration = launch + execute + sync.
#include <benchmark/benchmark.h>

#include "Redwood/Kernel.hpp"
#include "bench_common.hpp"

static void BM_BhKernel(benchmark::State& state) {
  const int leaf = static_cast<int>(state.range(0));
  const int active = static_cast<int>(state.range(1));
  constexpr dist::Gravity functor;

  bench::EnsureInit();
  bench::KernelInputs in;
  in.Alloc(/*leaves=*/1024, leaf, active, /*k=*/1, /*bodies=*/true);

  // Warm up.
  in.ResetOut(0.f);
  redwood::BarnesKernel(0, 0, in.u_lnt, leaf, in.u_q, in.u_node_idx, active,
                        in.u_out, functor);
  redwood::DeviceSynchronize();

  for (auto _ : state) {
    state.PauseTiming();
    in.ResetOut(0.f);  // BarnesKernel accumulates (+=); reset to a fixed start.
    state.ResumeTiming();

    redwood::BarnesKernel(0, 0, in.u_lnt, leaf, in.u_q, in.u_node_idx, active,
                          in.u_out, functor);
    redwood::DeviceSynchronize();
    benchmark::DoNotOptimize(in.u_out[0]);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * active *
                          leaf);
  in.Free();
}

BENCHMARK(BM_BhKernel)->REDWOOD_BENCH_ARGS;
