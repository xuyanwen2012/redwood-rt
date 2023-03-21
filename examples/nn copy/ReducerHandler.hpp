#pragma once

#include <algorithm>
#include <array>
#include <limits>

#include "../LeafNodeTable.hpp"
#include "../NnBuffer.hpp"
#include "AppParams.hpp"
#include "Redwood.hpp"
#include "Redwood/Kernels.hpp"

namespace rdc {

// Make this Compile time parameter
constexpr auto kNumThreads = 1;
constexpr auto kNumStreams = redwood::kNumStreams;

// ----- Others -----------

// Barnes-Hut Version
// Need to have a interface for Reducer in general
// And a specialized class for BH, NN, and KNN
struct ReducerHandler {
  void Init() {
    for (int i = 0; i < kNumStreams; ++i) {
      usm_buffers[i].Allocate(app_params.batch_size);

      // Initilize Results
      usm_result[i] = redwood::UsmMalloc<float>(app_params.batch_size);

      // CUDA Only
      redwood::AttachStreamMem(i, usm_buffers[i].query_point.data());
      redwood::AttachStreamMem(i, usm_buffers[i].leaf_idx.data());
      redwood::AttachStreamMem(i, usm_result[i]);
    }
  }

  void Release() {
    for (int i = 0; i < kNumStreams; ++i) {
      redwood::UsmFree(usm_result[i]);

      // Mannuelly free the vector
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
}

inline void ReleaseReducers() {
  rdc::FreeLeafNodeTalbe();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Release();
}

inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx, const Point4F q) {
  rhs[tid].UsmBuffer(stream_id).Push(q, node_idx);
}

inline void ClearBuffer(const int tid, const int stream_id) {
  rhs[tid].UsmBuffer(stream_id).Clear();
}

inline void ClearResult(const int tid, const int stream_id) {
  // std::fill_n(rhs[tid].usm_result[stream_id], app_params.batch_size,
  //             std::numeric_limits<float>::max());
}

_NODISCARD inline const int NextStream(const int stream_id) {
  return (kNumStreams - 1) - stream_id;
}

// Mostly for KNN
template <typename T>
_NODISCARD T* GetResultAddr(const int tid, const int stream_id) {
  return rhs[tid].UsmResultAddr(stream_id);
}

// Mostly for BH
template <typename T>
_NODISCARD T GetResultValueUnchecked(const int tid, const int stream_id) {
  return *GetResultAddr<T>(tid, stream_id);
}

inline float KernelFuncKnn(const Point4F p, const Point4F q) {
  auto dist = float();

  for (int i = 0; i < 4; ++i) {
    const auto diff = p.data[i] - q.data[i];
    dist += diff * diff;
  }

  return sqrtf(dist);
}

template <int LeafSize>
void Debug(const int* u_leaf_indices, const Point4F* u_q_points,
           const int num_active_leafs, float* outs, const Point4F* u_lnt_data,
           const int* u_lnt_sizes, int, int) {
  for (int i = 0; i < num_active_leafs; ++i) {
    const auto leaf_id_to_load = u_leaf_indices[i];
    const auto q = u_q_points[i];
    auto my_min = std::numeric_limits<float>::max();

    for (int j = 0; j < LeafSize; ++j) {
      const auto p = u_lnt_data[leaf_id_to_load * LeafSize + j];
      const auto dist = KernelFuncKnn(p, q);
      // std::cout << dist << std::endl;
      my_min = std::min(my_min, dist);
    }

    outs[i] = my_min;
    // std::cout << "outs[i]: " << outs[i] << std::endl;
    // std::cout << "my_min: " << my_min << std::endl;
    // exit(0);
  }
}

inline void LuanchKernelAsync(const int tid, const int stream_id) {
  // TODO: Need to select User's kernel

  Debug<32>(
      rhs[tid].UsmBuffer(stream_id).LData(), /* Buffered data to process */
      rhs[tid].UsmBuffer(stream_id).QData(), /* / */
      rhs[tid].UsmBuffer(stream_id).Size(),
      rhs[tid].UsmResultAddr(stream_id), /* Return Addr */
      rdc::LntDataAddr(),                /* Shared data */
      nullptr,                           /* Ignore for now */
      app_params.max_leaf_size, stream_id);
}

}  // namespace rdc
