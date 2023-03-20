#pragma once

// namespace redwood {
// enum class Backends {
//   kCpu = 0,
//   kCuda,
//   kSycl,
//   kDuet,
// };
// }

// #ifdef REDWOOD_BACKEND
// #if REDWOOD_BACKEND == 1
// // #define REDWOOD_IN_CUDA
// constexpr auto kRedwoodBackend = redwood::Backends::kCuda;
// #elif REDWOOD_BACKEND == 2
// constexpr auto kRedwoodBackend = redwood::Backends::kSycl;
// #elif REDWOOD_BACKEND == 3
// constexpr auto kRedwoodBackend = redwood::Backends::kDuet;
// #endif
// #else
// constexpr auto kRedwoodBackend = redwood::Backends::kCpu;
// #define REDWOOD_IN_CPU
// #endif

#define NO_OP 0

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

#define _MAYBE_UNUSED [[maybe_unused]]
