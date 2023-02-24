#pragma once

#include <chrono>
#include <iostream>
#include <random>

static float MyRand(float min = 0.0, float max = 1.0) {
  // 114514 and 233
  static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(generator);
}

template <typename Func>
void TimeTask(const std::string& task_name, Func&& f) {
  const auto t0 = std::chrono::high_resolution_clock::now();

  std::forward<Func>(f)();

  const auto t1 = std::chrono::high_resolution_clock::now();
  const auto time_span =
      std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0);

  std::cout << "Finished " << task_name << "! Time took: " << time_span.count()
            << "s. " << std::endl;
}
