cmake_minimum_required(VERSION 3.20)
project(redwood VERSION 1.0.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
# Which accelerator backend to build against. Each backend implements the same
# redwood:: API declared in include/Redwood/. The CPU backend is the portable
# default and the ground truth for the test suite (the homogeneous baseline
# from the Redwood paper).
set(REDWOOD_BACKEND "cpu" CACHE STRING "Accelerator backend: cpu | cuda | sycl")
set_property(CACHE REDWOOD_BACKEND PROPERTY STRINGS cpu cuda sycl)

option(REDWOOD_BUILD_TESTS "Build the unit test suite" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Shared include dir (the public redwood:: headers + functors).
set(REDWOOD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

message(STATUS "Redwood backend: ${REDWOOD_BACKEND}")

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
if(REDWOOD_BACKEND STREQUAL "cpu")
  add_subdirectory(accelerator/cpu)
elseif(REDWOOD_BACKEND STREQUAL "cuda")
  # Set the target architecture(s) BEFORE enabling the CUDA language, otherwise
  # enable_language() bakes in a default and the GPU actually present may have
  # no device code (causes "No device code available for GPU ISA NN" at runtime).
  # "native" detects the local GPU; override with -DCMAKE_CUDA_ARCHITECTURES=89.
  set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures (e.g. 89, or 'native')")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_subdirectory(accelerator/cuda)
elseif(REDWOOD_BACKEND STREQUAL "sycl")
  if(NOT CMAKE_CXX_COMPILER MATCHES "icpx|dpcpp|clang")
    message(WARNING "SYCL backend needs a SYCL compiler; configure with -DCMAKE_CXX_COMPILER=icpx")
  endif()
  add_subdirectory(accelerator/sycl)
else()
  message(FATAL_ERROR "Unknown REDWOOD_BACKEND='${REDWOOD_BACKEND}' (expected cpu|cuda|sycl).")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(REDWOOD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
