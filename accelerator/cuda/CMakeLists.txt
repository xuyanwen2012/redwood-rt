get_filename_component(BACKEND_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

set(LIBRARY_NAME redwood_${BACKEND_NAME})
message(STATUS "Add ${LIBRARY_NAME}")

enable_language(CXX CUDA)

# set(CMAKE_CUDA_STANDARD 14)

# set_source_files_properties(
#   Redwood.cu
#   PROPERTIES LANGUAGE CUDA
#  )

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3 -Xcompiler -Wextra)

# target_link_libraries(mainapp ${MYCUDALIB} ${CUDART_LIBRARY})

# cuda_add_library(${LIBRARY_NAME} Redwood.cu)

add_library(${LIBRARY_NAME} STATIC
 Redwood.cu
)

set_target_properties(${LIBRARY_NAME}
  PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${LIBRARY_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/accelerator/cuda/extern/cub-1.17.2
  ${PROJECT_SOURCE_DIR}/accelerator/cuda/extern/thrust-1.17.2
)
