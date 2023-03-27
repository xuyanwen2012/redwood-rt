get_filename_component(BACKEND_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

set(LIBRARY_NAME redwood_${BACKEND_NAME})
message(STATUS "Add ${LIBRARY_NAME}")

enable_language(CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

add_library(${LIBRARY_NAME} STATIC
  Redwood.cu
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/include
  # ${PROJECT_SOURCE_DIR}/accelerator/cuda/extern
)

# set(LANGUAGE_SET CXX CUDA)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -O3>)

set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 35 50 52 53 60 61 70 72 75)

set_source_files_properties(
  Redwood.cu
  PROPERTIES LANGUAGE CUDA
 )