# SYCL backend: implements the redwood:: API with SYCL USM (malloc_shared),
# queues (the async work queue), and a parallel_for NN reduction.
#
# Requires a SYCL compiler (Intel oneAPI icpx / dpcpp). Configure with:
#   -DCMAKE_CXX_COMPILER=icpx
# -fsycl is exported PUBLIC so the test/app executables that link this backend
# are compiled and linked as SYCL programs too (pulls in the SYCL runtime).
add_library(redwood_sycl STATIC
  Usm.cpp
  Core.cpp
  Kernel.cpp
)

target_include_directories(redwood_sycl PUBLIC ${REDWOOD_INCLUDE_DIR})
target_compile_features(redwood_sycl PUBLIC cxx_std_17)

target_compile_options(redwood_sycl PUBLIC -fsycl)
target_link_options(redwood_sycl PUBLIC -fsycl)

# Stable alias: consumers link redwood::backend regardless of backend choice.
add_library(redwood::backend ALIAS redwood_sycl)
