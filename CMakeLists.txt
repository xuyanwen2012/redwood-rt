cmake_minimum_required(VERSION 3.23)

project(redwood VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(-Wall -pedantic -Wextra)
  add_compile_options(-Werror=return-type)
  add_compile_options(-pthread)
else()
  message(SEND_ERROR "Unspported platform")
endif()

add_library(redwood_cpu STATIC
  accelerator/cpu/Redwood.cpp
)

target_include_directories(redwood_cpu PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_subdirectory(examples)