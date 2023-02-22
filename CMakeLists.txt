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

# e.g. "cpu;cpu1" no trailing semicolon
set(BACKENDS "" CACHE STRING "List of backends to compile")

if(BACKENDS STREQUAL "")
  message(FATAL_ERROR "Must specify at least one backend (e.g. -DBACKENDS=\"cpu;cpu1\")")
endif()

add_subdirectory(accelerator)
add_subdirectory(examples)
