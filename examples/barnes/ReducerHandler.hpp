#pragma once

#include <array>

#include "../IndicesBuffer.hpp"
#include "../LeafNodeTable.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Kernels.hpp"
#include "Redwood/Point.hpp"

namespace rdc {

constexpr auto kNumThreads = 1;
constexpr auto kNumStreams = redwood::kNumStreams;
constexpr auto kMaxLeafSize = 64;

// ----- Others -----------

// Barnes-Hut Version
// Need to have a interface for Reducer in general
// And a specialized class for BH, NN, and KNN
struct ReducerHandler {
  void Init(const int batch_size = 1024) {
    for (int i = 0; i < kNumStreams; ++i) {
      constexpr auto result_size = 1;
      usm_result[i] = redwood::UsmMalloc<float>(result_size);
      usm_buffers[i].Allocate(batch_size);

      // CUDA Only
      redwood::AttachStreamMem(i, usm_buffers[i].leaf_nodes.data());
      redwood::AttachStreamMem(i, usm_result[i]);
    }
  }

  void Release() {
    for (int i = 0; i < kNumStreams; ++i) {
      redwood::UsmFree(usm_result[i]);

      // Mannuelly free
      redwood::UsmVector<int> tmp;
      usm_buffers[i].leaf_nodes.swap(tmp);
    }
  }

  _NODISCARD IndicesBuffer& UsmBuffer(const int stream_id) {
    return usm_buffers[stream_id];
  }
  _NODISCARD float* UsmResultAddr(const int stream_id) {
    return usm_result[stream_id];
  }

  _NODISCARD const Point4F QueryPoint(const int stream_id) const {
    return h_query[stream_id];
  }

  void SetQueryPoint(const int stream_id, const Point4F q) {
    h_query[stream_id] = q;
  }

  // Point to Buffers
  std::array<IndicesBuffer, kNumStreams> usm_buffers;
  std::array<float*, kNumStreams> usm_result;
  std::array<Point4F, kNumStreams> h_query;
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

inline void SetQuery(const int tid, const int stream_id, const Point4F q) {
  rhs[tid].SetQueryPoint(stream_id, q);
};

inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx) {
  // You can push as many ints as you want.
  rhs[tid].UsmBuffer(stream_id).PushLeaf(node_idx);
};

inline void ReduceBranchNode(const int tid, const int stream_id,
                             const Point4F data){};

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
  redwood::ComputeOneBatchAsync(
      rhs[tid].UsmBuffer(stream_id).Data(), /* Buffered data to process */
      rhs[tid].UsmBuffer(stream_id).Size(), /* / */
      rhs[tid].UsmResultAddr(stream_id),    /* Return Addr */
      rdc::LntDataAddr(),                   /* Shared data */
      nullptr,                              /* Ignore for now */
      rhs[tid].QueryPoint(stream_id),       /* Single data */
      stream_id);
}

}  // namespace rdc
