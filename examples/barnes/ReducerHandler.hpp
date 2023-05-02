#pragma once

#include <array>
#include <utility>

#include "../Utils.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood.hpp"
#include "Redwood/Point.hpp"

namespace rdc {

// Shared accross threads, streams.
inline Point4F* lnt_base_addr = nullptr;
inline int* lnt_size_base_addr = nullptr;
inline int stored_max_leaf_size;

_NODISCARD inline std::pair<Point4F*, int*> AllocateLnt(
    const int num_leaf_nodes, const int max_leaf_size) {
  stored_max_leaf_size = max_leaf_size;

  lnt_base_addr = redwood::UsmMalloc<Point4F>(num_leaf_nodes * max_leaf_size);
  lnt_size_base_addr = redwood::UsmMalloc<int>(num_leaf_nodes);

  return std::make_pair(lnt_base_addr, lnt_size_base_addr);
}

_NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
  return lnt_base_addr + node_idx * stored_max_leaf_size;
}

_NODISCARD inline const int LntSizeAt(const int node_idx) {
  return lnt_size_base_addr[node_idx];
}

using IndicesBuffer = redwood::UsmVector<int>;

struct ResultBuffer {
  void Alloc() { underlying_dat = redwood::UsmMalloc<float>(1); }
  void DeAlloc() const { redwood::UsmFree(underlying_dat); }
  _NODISCARD float* GetAddrAt() const { return underlying_dat; }
  // 1 float for BH
  float* underlying_dat;
};

inline int stored_num_threads;
inline std::vector<std::array<IndicesBuffer, 2>> buffers;
inline std::vector<std::array<ResultBuffer, 2>> result_addr;

inline std::vector<std::array<Point4F, 2>> h_query;

inline void Init(const int num_thread, const int batch_size) {
  redwood::Init(num_thread);
  stored_num_threads = num_thread;

  buffers.resize(num_thread);
  result_addr.resize(num_thread);
  h_query.resize(num_thread);
  for (int tid = 0; tid < num_thread; ++tid) {
    for (int i = 0; i < 2; ++i) {
      // Unified Shared Memory
      buffers[tid][i].reserve(batch_size);
      result_addr[tid][i].Alloc();

      redwood::AttachStreamMem(tid, i, buffers[tid][i].data());
      redwood::AttachStreamMem(tid, i, result_addr[tid][i].underlying_dat);
    }
  }
}

inline void Release() {
  for (int tid = 0; tid < stored_num_threads; ++tid) {
    for (int i = 0; i < 2; ++i) {
      // redwood::UsmFree(result_addr[tid][i]);
      result_addr[tid][i].DeAlloc();

      // Mannuelly free a std::vector
      IndicesBuffer tmp;
      buffers[tid][i].swap(tmp);
    }
  }

  redwood::UsmFree(lnt_base_addr);
  redwood::UsmFree(lnt_size_base_addr);
}

inline void ResetBuffer(const int tid, const int cur_stream) {
  buffers[tid][cur_stream].clear();
}

_NODISCARD inline float* RequestResultAddr(const int tid, const int stream_id) {
  return result_addr[tid][stream_id].GetAddrAt();
}

inline void SetQuery(const int tid, const int stream_id, const Point4F q) {
  h_query[tid][stream_id] = q;
}

inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx) {
  // buffers[tid][stream_id].push_back(node_idx);
  auto my_sum = float();
  dist::Gravity functor{};

  const auto node_addr = LntDataAddrAt(node_idx);
  const auto num_leaf = lnt_size_base_addr[node_idx];
  for (int j = 0; j < num_leaf; ++j) {
    my_sum += functor(h_query[tid][stream_id], node_addr[j]);
  }

  *result_addr[tid][stream_id].underlying_dat += my_sum;
}

// inline void DebugCpuReduction(const IndicesBuffer& buf,
//                               const dist::Gravity functor, const Point4F q,
//                               float* result_addr) {
//   const auto n = buf.size();
//   auto my_sum = float();

//   // i is batch id, = tid, = index in the buffer
//   for (int i = 0; i < n; ++i) {
//     const auto node_idx = buf[i];
//     const auto node_addr = LntDataAddrAt(node_idx);
//     for (int j = 0; j < stored_max_leaf_size; ++j) {
//       my_sum += functor(node_addr[j], q);
//     }
//   }

//   *result_addr += my_sum;
// }

inline void LaunchAsyncWorkQueue(const int tid, const int stream_id) {
  const auto num_active = buffers[tid][stream_id].size();

  if constexpr (kDebugMod) {
    std::cout << "rdc::LaunchAsyncWorkQueue "
              << "tid: " << tid << ", stream: " << stream_id << ", "
              << buffers[tid][stream_id].size() << " actives." << std::endl;
    // 128? 256?
  }

  // dist::Gravity functor{};
  // DebugCpuReduction(buffers[tid][stream_id], functor,
  // h_query[tid][stream_id],
  //                   result_addr[tid][stream_id]);

  // redwood::NearestNeighborKernel(
  //     tid, stream_id, lnt_base_addr, stored_max_leaf_size,
  //     buffers[tid][stream_id].u_qs, buffers[tid][stream_id].u_leaf_idx,
  //     num_active, result_addr[tid][stream_id].underlying_dat, functor);
}
}  // namespace rdc
