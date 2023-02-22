get_filename_component(BACKEND_NAME "${CMAKE_CURRENT_SOURCE_DIR}" NAME)

set(LIBRARY_NAME redwood_${BACKEND_NAME})

message(STATUS "Add ${LIBRARY_NAME}")

add_library(${LIBRARY_NAME} STATIC
  Redwood.cpp
)

target_include_directories(${LIBRARY_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/include
)
