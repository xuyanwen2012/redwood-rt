#!/usr/bin/env bash
# Run the kernel benchmarks across all available backends and collect JSON.
#
# CPU and CUDA run natively; SYCL runs in the oneAPI container via
# tools/sycl-docker.sh. Results land in ./results/<backend>_<algo>.json with
# aggregated stats (mean/median/stddev/cv) over REPS repetitions.
#
# Usage:
#   tools/run-benchmarks.sh                # cpu + cuda + sycl (whatever is available)
#   BACKENDS="cpu cuda" tools/run-benchmarks.sh
#   REPS=20 tools/run-benchmarks.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
BACKENDS="${BACKENDS:-cpu cuda sycl}"
REPS="${REPS:-10}"
ALGOS="nn bh knn"
GBENCH_FLAGS="--benchmark_repetitions=$REPS --benchmark_report_aggregates_only=true"
mkdir -p results

run_native() {  # $1 = backend, $2 = build dir, $3.. = extra cmake args
  local backend="$1" dir="$2"; shift 2
  echo "==================== $backend ===================="
  cmake -S . -B "$dir" -DREDWOOD_BACKEND="$backend" -DREDWOOD_BUILD_BENCHMARKS=ON "$@" >/dev/null
  cmake --build "$dir" -j >/dev/null
  for a in $ALGOS; do
    ./"$dir"/benchmarks/bench_${a}_kernel $GBENCH_FLAGS \
      --benchmark_format=json --benchmark_out="results/${backend}_${a}.json"
  done
}

for backend in $BACKENDS; do
  case "$backend" in
    cpu)  run_native cpu  build ;;
    cuda) run_native cuda build-cuda -DCMAKE_CUDA_ARCHITECTURES=native ;;
    sycl) echo "==================== sycl (container) ===================="
          REPS="$REPS" "$REPO_ROOT/tools/sycl-docker.sh" bench ;;
    *)    echo "unknown backend: $backend" >&2; exit 1 ;;
  esac
done

echo
echo "JSON results written to: $REPO_ROOT/results/"
ls -1 results/*.json 2>/dev/null || true
