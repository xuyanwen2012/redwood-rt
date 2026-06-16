# CPU reference backend: a pure-host implementation of the redwood:: API.
# It needs no GPU/FPGA toolchain, which makes it the ground-truth backend for
# the test suite and the homogeneous baseline described in the Redwood paper.
add_library(redwood_cpu STATIC
  Usm.cpp
  Core.cpp
  Kernel.cpp
)

target_include_directories(redwood_cpu PUBLIC ${REDWOOD_INCLUDE_DIR})
target_compile_features(redwood_cpu PUBLIC cxx_std_17)

# Expose a stable alias so consumers can link redwood::backend regardless of
# which concrete backend was selected.
add_library(redwood::backend ALIAS redwood_cpu)
