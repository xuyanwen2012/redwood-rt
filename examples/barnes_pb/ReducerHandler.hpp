#pragma once

#include <array>

#include "../IndicesBuffer.hpp"
#include "../LeafNodeTable.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Kernels.hpp"
#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"

namespace rdc {

constexpr auto kNumThreads = 1;
constexpr auto kNumStreams = redwood::kNumStreams;
constexpr auto kMaxLeafSize = 64;
constexpr auto kPointBlockingSize = 128;

// ----- Others -----------

// Barnes-Hut Version
// Need to have a interface for Reducer in general
// And a specialized class for BH, NN, and KNN
struct ReducerHandler {
  void Init(const int batch_size = 1024) {
    for (int i = 0; i < kNumStreams; ++i) {
      // malloc sizeof(float) * 128
      usm_result[i] = redwood::UsmMalloc<float>(kPointBlockingSize);
      redwood::AttachStreamMem(i, usm_result[i]);

      // malloc sizeof(Point4F) * 128
      usm_query_points[i] = redwood::UsmMalloc<Point4F>(kPointBlockingSize);
      redwood::AttachStreamMem(i, usm_query_points[i]);

      for (int j = 0; j < kPointBlockingSize; ++j) {
        usm_buffers[i][j].Allocate(batch_size);
        redwood::AttachStreamMem(i, usm_buffers[i][j].leaf_nodes.data());
      }
    }
  }

  void Release() {
    for (int i = 0; i < kNumStreams; ++i) {
      redwood::UsmFree(usm_result[i]);
      redwood::UsmFree(usm_query_points[i]);

      // Mannuelly free
      redwood::UsmVector<int> tmp;
      for (int j = 0; j < kPointBlockingSize; ++j) {
        usm_buffers[i][j].leaf_nodes.swap(tmp);
      }
    }
  }

  _NODISCARD IndicesBuffer& UsmBuffer(const int stream_id, const int pb_idx) {
    return usm_buffers[stream_id][pb_idx];
  }

  _NODISCARD float* UsmResultAddr(const int stream_id) {
    return usm_result[stream_id];
  }

  // Return a single 'q'
  _NODISCARD const Point4F QueryPoint(const int stream_id,
                                      const int pb_idx) const {
    return usm_query_points[stream_id][pb_idx];
  }

  // Return all 'q'
  _NODISCARD const Point4F* QueryPoints(const int stream_id) const {
    return usm_query_points[stream_id];
  }

  void SetQueryPoint(const int stream_id, const int pb_idx, const Point4F& q) {
    usm_query_points[stream_id][pb_idx] = q;
  }

  // std::array<IndicesBuffer, kNumStreams> usm_buffers;

  // num_stream * pb_size * batch_size
  // = 2 x 128 x 1024
  // 256 pointers pointing to regions of 1024 USM
  // This sucks but in future we need a better (more dense) buffer
  std::array<std::array<IndicesBuffer, kPointBlockingSize>, kNumStreams>
      usm_buffers;

  // Lets make point blocking size a constant. assuming 128
  std::array<float*, kNumStreams> usm_result;

  // Every q need to have 1024 ints (leaf node indices)
  Point4F q;

  // // 1024 size buffer
  // // IndicesBuffer
  //  redwood::UsmVector<int>

  std::array<Point4F, 2> qs;  // 2 q s

  // redwood::UsmVector<int>
  // malloc sizeof(Point4F) * 128
  // Point4F* q;

  std::array<Point4F*, kNumStreams> usm_query_points;
};

inline std::array<ReducerHandler, kNumThreads> rhs;

inline void InitReducers() {
  redwood::Init();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Init();
}

inline void ReleaseReducers() {
  rdc::FreeLeafNodeTalbe();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Release();
}

// -------------------- Yanwen's addition Begin ----------------------------
// Note: temporary solution. I will futhur refactor this at the end.
// --------------------------------------------------------------------------

inline void SetQuery_PB(const int tid, const int stream_id, const int pb_idx,
                        const Point4F& q) {
  rhs[tid].SetQueryPoint(stream_id, pb_idx, q);
}

inline void ReduceLeafNode_PB(const int tid, const int stream_id,
                              const int pb_idx, const int node_idx) {
  rhs[tid].UsmBuffer(stream_id, pb_idx).PushLeaf(node_idx);
  // rhs[tid].UsmBuffer(stream_id).PushLeaf(node_idx);
}

inline void LuanchKernelAsync_PB(const int tid, const int stream_id) {
  // Just do some printing to see if data are collected correctly
}

template <typename T>
_NODISCARD T GetResultValueUnchecked_PB(const int tid, const int stream_id,
                                        const int pb_idx) {}

// -------------------- Yanwen's addition End ----------------------------

// inline void SetQuery(const int tid, const int stream_id, const Point4F q) {
//   rhs[tid].SetQueryPoint(stream_id, q);
// }

// inline void ReduceLeafNode(const int tid, const int stream_id,
//                            const int node_idx) {
//   // You can push as many ints as you want.
//   rhs[tid].UsmBuffer(stream_id).PushLeaf(node_idx);
// }

// inline void ReduceBranchNode(const int tid, const int stream_id,
//                              const Point4F data){};

// inline void ClearBuffer(const int tid, const int stream_id) {
//   rhs[tid].UsmBuffer(stream_id).Clear();
// }

// _NODISCARD inline int NextStream(const int stream_id) {
//   return (kNumStreams - 1) - stream_id;
// }

// // Mostly for KNN
// template <typename T>
// _NODISCARD T* GetResultAddr(const int tid, const int stream_id) {
//   return rhs[tid].UsmResultAddr(stream_id);
// }

// // Mostly for BH/NN
// template <typename T>
// _NODISCARD T GetResultValueUnchecked(const int tid, const int stream_id) {
//   return *GetResultAddr<T>(tid, stream_id);
// }

// inline void LuanchKernelAsync(const int tid, const int stream_id) {
//   // TODO: Need to select User's kernel
//   redwood::ComputeOneBatchAsync(
//       rhs[tid].UsmBuffer(stream_id).Data(), /* Buffered data to process */
//       static_cast<int>(rhs[tid].UsmBuffer(stream_id).Size()), /* / */
//       rhs[tid].UsmResultAddr(stream_id),                      /* Return Addr
//       */ rdc::LntDataAddr(),                                     /* Shared
//       data */ nullptr,                        /* Ignore for now */
//       rhs[tid].QueryPoint(stream_id), /* Single data */
//       stream_id);
// }

}  // namespace rdc
