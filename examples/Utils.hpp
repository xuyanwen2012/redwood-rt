#pragma once

#include <chrono>
#include <iostream>

#define NO_OP 0

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

#ifndef _HAS_NODISCARD
#ifndef __has_cpp_attribute
#define _HAS_NODISCARD 0
#elif __has_cpp_attribute(nodiscard) >= \
    201603L  // TRANSITION, VSO#939899 (need toolset update)
#define _HAS_NODISCARD 1
#else
#define _HAS_NODISCARD 0
#endif
#endif  // _HAS_NODISCARD

#if _HAS_NODISCARD
#define _NODISCARD [[nodiscard]]
#else  // ^^^ CAN HAZ [[nodiscard]] / NO CAN HAZ [[nodiscard]] vvv
#define _NODISCARD
#endif  // _HAS_NODISCARD