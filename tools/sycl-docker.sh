#!/usr/bin/env bash
# Build and run the redwood test suite against the SYCL backend inside the
# Intel oneAPI container (icpx). SYCL needs the oneAPI compiler + runtime, which
# we don't install on the host, so everything runs in the container. The local
# i9-14900K is used via the OpenCL CPU device (no GPU SYCL device required).
#
# Usage:
#   tools/sycl-docker.sh            # configure + build + ctest
#   tools/sycl-docker.sh build      # configure + build only
#
# Produces an out-of-source build dir ./build-sycl (gitignored). Runs as the
# host user so build artifacts are not owned by root.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${ONEAPI_IMAGE:-intel/oneapi-toolkit:latest}"
MODE="${1:-test}"

CMDS='set -e
cmake -S . -B build-sycl -DREDWOOD_BACKEND=sycl -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build-sycl -j'
if [ "$MODE" != "build" ]; then
  CMDS="$CMDS
ctest --test-dir build-sycl --output-on-failure"
fi

exec docker run --rm \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$REPO_ROOT:/work" -w /work \
  "$IMAGE" bash -lc "$CMDS"
