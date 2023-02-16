#pragma once

#include <array>

#include "../LeafNodeTable.hpp"
#include "../NnBuffer.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Kernels.hpp"
#include "Redwood/Point.hpp"

namespace rdc {

// Make this Compile time parameter
constexpr auto kNumThreads = 1;

constexpr auto kNumStreams = redwood::kNumStreams;
constexpr auto kMaxLeafSize = 64;
constexpr auto kK = 32;

// ----- Others -----------

// Barnes-Hut Version
// Need to have a interface for Reducer in general
// And a specialized class for BH, NN, and KNN
struct ReducerHandler {
  void Init(const int batch_size = 1024) {
    for (int i = 0; i < kNumStreams; ++i) {
      // Each batch need a block of 32 address.
      // thus, 1024 * 32 = 32768 bytes per stream
      usm_result[i] = redwood::UsmMalloc<float>(batch_size * kK);
      usm_buffers[i].Allocate(batch_size);

      // CUDA Only
      redwood::AttachStreamMem(i, usm_buffers[i].query_point.data());
      redwood::AttachStreamMem(i, usm_buffers[i].leaf_idx.data());
      redwood::AttachStreamMem(i, usm_result[i]);
    }
  }

  void Release() {
    for (int i = 0; i < kNumStreams; ++i) {
      redwood::UsmFree(usm_result[i]);

      // Mannuelly free
      redwood::UsmVector<Point4F> tmp;
      usm_buffers[i].query_point.swap(tmp);
      redwood::UsmVector<int> tmp2;
      usm_buffers[i].leaf_idx.swap(tmp2);
    }
  }

  _NODISCARD NnBuffer<Point4F>& UsmBuffer(const int stream_id) {
    return usm_buffers[stream_id];
  }

  _NODISCARD float* UsmResultAddr(const int stream_id) {
    return usm_result[stream_id];
  }

  // Point to Buffers
  std::array<NnBuffer<Point4F>, kNumStreams> usm_buffers;
  std::array<float*, kNumStreams> usm_result;
};

inline std::array<ReducerHandler, kNumThreads> rhs;

inline void InitReducers() {
  redwood::Init();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Init();
};

inline void ReleaseReducers() {
  rdc::FreeLeafNodeTalbe();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Release();
}

inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx, const Point4F q) {
  rhs[tid].UsmBuffer(stream_id).Push(q, node_idx);
};

inline void ClearBuffer(const int tid, const int stream_id) {
  rhs[tid].UsmBuffer(stream_id).Clear();
};

_NODISCARD inline const int NextStream(const int stream_id) {
  return (kNumStreams - 1) - stream_id;
};

// Mostly for KNN
template <typename T>
_NODISCARD T* GetResultAddr(const int tid, const int stream_id) {
  return rhs[tid].UsmResultAddr(stream_id);
}

// Mostly for BH/NN
template <typename T>
_NODISCARD T GetResultValueUnchecked(const int tid, const int stream_id) {
  return *GetResultAddr<T>(tid, stream_id);
}

inline void LuanchKernelAsync(const int tid, const int stream_id) {
  // TODO: Need to select User's kernel
  redwood::ProcessKnnAsync(
      rhs[tid].UsmBuffer(stream_id).LData(), /* Buffered data to process */
      rhs[tid].UsmBuffer(stream_id).QData(), /* / */
      rhs[tid].UsmBuffer(stream_id).Size(),
      rhs[tid].UsmResultAddr(stream_id), /* Return Addr */
      rdc::LntDataAddr(),                /* Shared data */
      nullptr,                           /* Ignore for now */
      stream_id);
}

}  // namespace rdc
