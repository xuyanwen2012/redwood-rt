#!/usr/bin/env bash
# Build and run the redwood suite against the SYCL backend inside the Intel
# oneAPI container (icpx). SYCL needs the oneAPI compiler + runtime, which we
# don't install on the host, so everything runs in the container.
#
# Devices: the container sees the i9-14900K via the OpenCL CPU device. The
# /dev/dri passthrough below also exposes an Intel iGPU *if one is enabled in
# BIOS* (the backend uses default_selector_v, so it would auto-select the GPU).
#
# Usage:
#   tools/sycl-docker.sh            # configure + build + ctest (correctness)
#   tools/sycl-docker.sh build      # configure + build only
#   tools/sycl-docker.sh bench      # configure (+benchmarks) + build + run benchmarks -> results/sycl_*.json
#
# Produces out-of-source build dir ./build-sycl (gitignored). Runs as the host
# user so artifacts are not owned by root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ONEAPI_IMAGE:-intel/oneapi-toolkit:latest}"
MODE="${1:-test}"
REPS="${REPS:-10}"

CONFIGURE='cmake -S . -B build-sycl -DREDWOOD_BACKEND=sycl -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release'
case "$MODE" in
  build)
    CMDS="set -e
$CONFIGURE
cmake --build build-sycl -j" ;;
  bench)
    CMDS="set -e
$CONFIGURE -DREDWOOD_BUILD_BENCHMARKS=ON
cmake --build build-sycl -j
mkdir -p results
for a in nn bh knn; do
  ./build-sycl/benchmarks/bench_\${a}_kernel \
    --benchmark_repetitions=$REPS --benchmark_report_aggregates_only=true \
    --benchmark_format=json --benchmark_out=results/sycl_\${a}.json
done" ;;
  *)
    CMDS="set -e
$CONFIGURE
cmake --build build-sycl -j
ctest --test-dir build-sycl --output-on-failure" ;;
esac

# Expose the render node (Intel iGPU) if present; harmless for the CPU device.
DRI_FLAGS=()
if [ -e /dev/dri ]; then
  DRI_FLAGS+=(--device /dev/dri)
  render_gid="$(getent group render | cut -d: -f3 || true)"
  [ -n "$render_gid" ] && DRI_FLAGS+=(--group-add "$render_gid")
fi

exec docker run --rm \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  "${DRI_FLAGS[@]}" \
  -v "$REPO_ROOT:/work" -w /work \
  "$IMAGE" bash -lc "$CMDS"
