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
  std::cout<<"allocate lnt"<<std::endl;
  lnt_size_base_addr = redwood::UsmMalloc<int>(num_leaf_nodes);
  return std::make_pair(lnt_base_addr, lnt_size_base_addr);
}

_NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
  return lnt_base_addr + node_idx * stored_max_leaf_size;
}

using IndicesBuffer = redwood::UsmVector<int>;

inline int stored_num_threads;
inline std::vector<std::array<IndicesBuffer, 2>> buffers;
inline std::vector<std::array<float*, 2>> result_addr;
inline std::vector<std::array<Point4F, 2>> h_query;
inline std::vector<std::array<float, 2>> h_br_result;

inline void Init(const int num_thread, const int batch_size) {
  std::cout<<"init.."<<std::endl;
  redwood::Init(num_thread);
      std::cout<<"finished init"<<std::endl;

  stored_num_threads = num_thread;

  buffers.resize(num_thread);
  result_addr.resize(num_thread);
  h_query.resize(num_thread);
  h_br_result.resize(num_thread);

  for (int tid = 0; tid < num_thread; ++tid) {
    for (int i = 0; i < 2; ++i) {
      // Unified Shared Memory
      buffers[tid][i].reserve(batch_size);
      result_addr[tid][i] = redwood::UsmMalloc<float>(1);

      redwood::AttachStreamMem(tid, i, buffers[tid][i].data());
      redwood::AttachStreamMem(tid, i, result_addr[tid][i]);
    }
  }
}

inline void Release() {
  for (int tid = 0; tid < stored_num_threads; ++tid) {
    for (int i = 0; i < 2; ++i) {
      redwood::UsmFree(result_addr[tid][i]);

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
  // Reset accumulator
  *result_addr[tid][cur_stream] = 0.0f;
  h_br_result[tid][cur_stream] = 0.0f;
}

_NODISCARD inline float GetResultValue(const int tid, const int stream_id) {
  const auto device_result = *result_addr[tid][stream_id];
  const auto host_result = h_br_result[tid][stream_id];
  return device_result + host_result;
}

inline void SetQuery(const int tid, const int stream_id, const Point4F q) {
  h_query[tid][stream_id] = q;
}

inline void ReduceBranchNode(const int tid, const int stream_id,
                             const Point4F center_of_mass) {
  constexpr dist::Gravity my_functor;
  my_functor(h_query[tid][stream_id], center_of_mass);
}

inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx) {
  buffers[tid][stream_id].push_back(node_idx);
}

// inline void DebugCpuReduction(const Buffer& buf, const dist::Euclidean
// functor,
//                               const ResultBuffer& results) {
//   const auto n = buf.Size();

//   // i is batch id, = tid, = index in the buffer
//   for (int i = 0; i < n; ++i) {
//     const auto node_idx = buf.u_leaf_idx[i];
//     const auto q = buf.u_qs[i];

//     const auto node_addr = LntDataAddrAt(node_idx);
//     const auto addr = reinterpret_cast<KnnSet<float,
//     1>*>(results.GetAddrAt(i));

//     for (int j = 0; j < stored_max_leaf_size; ++j) {
//       const float dist = functor(node_addr[j], q);
//       addr->Insert(dist);
//     }
//   }
// }

inline void LaunchAsyncWorkQueue(const int tid, const int stream_id) {
  const auto num_active = buffers[tid][stream_id].size();

  if constexpr (kDebugMod) {
    std::cout << "rdc::LaunchAsyncWorkQueue "
              << "tid: " << tid << ", stream: " << stream_id << ", "
              << buffers[tid][stream_id].size() << " actives." << std::endl;
    // 128? 256?
  }

  dist::Gravity functor{};
  // redwood::NearestNeighborKernel(
  //     tid, stream_id, lnt_base_addr, stored_max_leaf_size,
  //     buffers[tid][stream_id].u_qs, buffers[tid][stream_id].u_leaf_idx,
  //     num_active, result_addr[tid][stream_id].underlying_dat, functor);
}
}  // namespace rdc
