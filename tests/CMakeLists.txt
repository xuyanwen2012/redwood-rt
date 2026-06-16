# GoogleTest is fetched at configure time so the suite builds with no manual
# setup (the old tests/Makefile hard-coded ~/googletest paths).
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
# Don't let GoogleTest override our compiler/linker settings on Windows.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# One executable per test file. Each links the selected redwood backend, so the
# very same tests can later be run against CUDA/SYCL by reconfiguring with
# -DREDWOOD_BACKEND=...
function(redwood_add_test name)
  add_executable(${name} ${name}.cpp)
  target_include_directories(${name} PRIVATE ${REDWOOD_INCLUDE_DIR})
  target_link_libraries(${name} PRIVATE redwood::backend GTest::gtest_main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

redwood_add_test(test_point)
redwood_add_test(test_knnset)
redwood_add_test(test_nn_kernel)
redwood_add_test(test_nn_traversal)
redwood_add_test(test_knn_reduction)
redwood_add_test(test_bh_reduction)
