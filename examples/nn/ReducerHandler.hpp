#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>

#include "../LeafNodeTable.hpp"
#include "../Utils.hpp"
#include "AppParams.hpp"
#include "Redwood/Kernels.hpp"
#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"

namespace rdc {

// Make this Compile time parameter
constexpr auto kNumThreads = 1;
constexpr auto kNumStreams = 2;

// ----- Others -----------

// Barnes-Hut Version
// Need to have a interface for Reducer in general
// And a specialized class for BH, NN, and KNN
template <typename T>
struct ReducerHandler {
  void Init() {
    for (int i = 0; i < kNumStreams; ++i) {
      // usm_leaf_idx[i] = redwood::UsmMalloc<int>(app_params.batch_size);
      // usm_query_point[i] = redwood::UsmMalloc<T>(app_params.batch_size);

      usm_leaf_idx[i].reserve(app_params.batch_size);
      usm_query_point[i].reserve(app_params.batch_size);
      usm_result[i] = redwood::UsmMalloc<float>(app_params.batch_size);

      // CUDA Only
      redwood::AttachStreamMem(i, usm_leaf_idx[i].data());
      redwood::AttachStreamMem(i, usm_query_point[i].data());
      redwood::AttachStreamMem(i, usm_result[i]);
    }
  }

  void Release() {
    for (int i = 0; i < kNumStreams; ++i) {
      // redwood::UsmFree(usm_leaf_idx[i]);
      // redwood::UsmFree(usm_query_point[i]);
      redwood::UsmFree(usm_result[i]);
    }
  }

  _NODISCARD float* UsmResultAddr(const int stream_id) {
    return usm_result[stream_id];
  }

  // std::array<int, kNumStreams> num_actives;
  // std::array<int*, kNumStreams> usm_leaf_idx;
  // std::array<T*, kNumStreams> usm_query_point;

  std::array<redwood::UsmVector<int>, kNumStreams> usm_leaf_idx;
  std::array<redwood::UsmVector<T>, kNumStreams> usm_query_point;

  // In BH, this is a single result (T)
  // In NN, this is (batch_size * T)
  // In KNN, this is (k * batch_size * T)
  std::array<float*, kNumStreams> usm_result;
};

inline std::array<ReducerHandler<Point4F>, kNumThreads> rhs;

inline void InitReducers() {
  // redwood::Init();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Init();
}

inline void ReleaseReducers() {
  rdc::FreeLeafNodeTalbe();
  for (int i = 0; i < kNumThreads; ++i) rhs[i].Release();
}

inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx, const Point4F& q) {
  rhs[tid].usm_leaf_idx[stream_id].push_back(node_idx);
  rhs[tid].usm_query_point[stream_id].push_back(q);

  // const auto cur = rhs[tid].num_actives[stream_id];

  // rhs[tid].usm_leaf_idx[stream_id][cur] = node_idx;
  // rhs[tid].usm_query_point[stream_id][cur] = q;

  // // increment
  // rhs[tid].num_actives[stream_id] = cur + 1;
}

inline void ClearBuffer(const int tid, const int stream_id) {
  rhs[tid].usm_leaf_idx[stream_id].clear();
  rhs[tid].usm_query_point[stream_id].clear();

  // rhs[tid].num_actives[stream_id] = 0;
  // rhs[tid].num_actives[stream_id] = 0;
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

void Debug(const int* u_leaf_indices, const Point4F* u_q_points,
           const int num_active_leafs, float* outs, const Point4F* u_lnt_data,
           const int* u_lnt_sizes, const int max_leaf_size, int) {
  for (int i = 0; i < num_active_leafs; ++i) {
    const auto leaf_id_to_load = u_leaf_indices[i];
    const auto q = u_q_points[i];
    auto my_min = std::numeric_limits<float>::max();

    for (int j = 0; j < max_leaf_size; ++j) {
      const auto p = u_lnt_data[leaf_id_to_load * max_leaf_size + j];
      const auto dist = KernelFuncKnn(p, q);
      my_min = std::min(my_min, dist);
    }

    outs[i] = std::min(outs[i], my_min);
  }
}

inline void LuanchKernelAsync(const int tid, const int stream_id) {
  // std::cout << "[stream " << stream_id << "] LuanchKernelAsync  "
  // << rhs[tid].num_actives[stream_id] << " items in buffer."
  // << std::endl;

  // redwood::ProcessNnAsync(rhs[tid].usm_leaf_idx[stream_id],     //
  //                         rhs[tid].usm_query_point[stream_id],  //
  //                         rhs[tid].num_actives[stream_id],      //
  //                         rhs[tid].usm_result[stream_id],       //
  //                         rdc::LntDataAddr(),        /* Shared data */
  //                         nullptr,                   /* Ignore for now */
  //                         app_params.max_leaf_size,  //
  //                         stream_id);

  // TODO: Need to select User's kernel
  Debug(rhs[tid].usm_leaf_idx[stream_id].data(),     //
        rhs[tid].usm_query_point[stream_id].data(),  //
        rhs[tid].usm_leaf_idx[stream_id].size(),     //
        rhs[tid].usm_result[stream_id],              //
        rdc::LntDataAddr(),                          /* Shared data */
        nullptr,                                     /* Ignore for now */
        app_params.max_leaf_size,                    //
        stream_id);
}

}  // namespace rdc
