# CUDA backend: implements the redwood:: API with CUDA managed memory (USM),
# streams (the async work queue), and a warp-reduction NN kernel.
#
# Note: this uses the CUB shipped with the CUDA Toolkit, not the vendored
# extern/cub-1.17.2 tree (which predates modern CUDA and no longer compiles).
add_library(redwood_cuda STATIC
  Usm.cu
  Core.cu
  Kernel.cu
)

target_include_directories(redwood_cuda PUBLIC ${REDWOOD_INCLUDE_DIR})
target_compile_features(redwood_cuda PUBLIC cxx_std_17)

set_target_properties(redwood_cuda PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)

target_compile_options(redwood_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xptxas -O3>)

# cudart so the C++ test/app executables that link this static lib resolve the
# CUDA runtime symbols.
target_link_libraries(redwood_cuda PUBLIC CUDA::cudart)

# Stable alias: consumers link redwood::backend regardless of backend choice.
add_library(redwood::backend ALIAS redwood_cuda)
