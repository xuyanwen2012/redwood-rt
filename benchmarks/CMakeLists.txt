# Google Benchmark is fetched at configure time (mirrors the GoogleTest setup in
# tests/). Disable its tests/install/Werror so it builds cleanly under g++, nvcc
# host, and icpx.
include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
)
FetchContent_MakeAvailable(googlebenchmark)

# Google Benchmark compiles its own sources with -pedantic-errors, which makes
# very new clang/icpx (2026) treat benchmark.h's __COUNTER__ use as a hard
# -Wc2y-extensions error. -Wno-error can't undo -pedantic-errors, so disable
# that specific pedantic warning (last flag wins). Guarded to Clang/IntelLLVM so
# g++/nvcc-host builds are untouched.
foreach(t benchmark benchmark_main)
  if(TARGET ${t})
    target_compile_options(${t} PRIVATE
      $<$<CXX_COMPILER_ID:Clang,IntelLLVM>:-Wno-c2y-extensions>)
  endif()
endforeach()

# One executable per benchmark file. Each links the selected redwood backend, so
# the same benchmark runs against CPU/CUDA/SYCL by reconfiguring
# -DREDWOOD_BACKEND=...   (mirrors redwood_add_test in tests/CMakeLists.txt)
function(redwood_add_benchmark name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE
    ${REDWOOD_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${name} PRIVATE redwood::backend benchmark::benchmark_main)
endfunction()

redwood_add_benchmark(bench_nn_kernel)
redwood_add_benchmark(bench_bh_kernel)
redwood_add_benchmark(bench_knn_kernel)
