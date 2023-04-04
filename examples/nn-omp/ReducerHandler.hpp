#pragma once

#include <array>

#include "../Utils.hpp"
#include "../nn/KnnSet.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "Redwood/Kernel.hpp"
#include "Redwood/Point.hpp"

using Task = std::pair<int, Point4F>;

#include "Redwood.hpp"

namespace rdc {

// Shared accross threads, streams.
inline Point4F* lnt_base_addr = nullptr;
inline int stored_max_leaf_size;

_NODISCARD inline Point4F* AllocateLnt(const int num_leaf_nodes,
                                       const int max_leaf_size) {
  stored_max_leaf_size = max_leaf_size;
  lnt_base_addr = redwood::UsmMalloc<Point4F>(num_leaf_nodes * max_leaf_size);
  return lnt_base_addr;
}

_NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
  return lnt_base_addr + node_idx * stored_max_leaf_size;
}

// For NN and KNN
struct Buffer {
  void Alloc(const int buffer_size) {
    u_qs = redwood::UsmMalloc<Point4F>(buffer_size);
    u_leaf_idx = redwood::UsmMalloc<int>(buffer_size);
  }

  void DeAlloc() const {
    redwood::UsmFree(u_qs);
    redwood::UsmFree(u_leaf_idx);
  }

  _NODISCARD int Size() const { return num_active; }

  void Reset() { num_active = 0; }

  void Push(const Task& task, const int node_idx) {
    u_qs[num_active] = task.second;
    u_leaf_idx[num_active] = node_idx;
    ++num_active;
  }

  int num_active;
  Point4F* u_qs;
  int* u_leaf_idx;
};

struct ResultBuffer {
  void Alloc(const int buffer_size, const int k = 1) {
    stored_k = k;
    underlying_dat = redwood::UsmMalloc<float>(buffer_size * k);
  }

  void DeAlloc() const { redwood::UsmFree(underlying_dat); }

  _NODISCARD float* GetAddrAt(const int executor_id) const {
    return underlying_dat + executor_id * stored_k;
  }

  float* underlying_dat;
  int stored_k;
};

inline int stored_num_threads;
inline std::vector<std::array<Buffer, 2>> buffers;
inline std::vector<std::array<ResultBuffer, 2>> result_addr;

inline void Init(const int num_thread, const int batch_size) {
  redwood::Init(num_thread);
  stored_num_threads = num_thread;

  buffers.resize(num_thread);
  result_addr.resize(num_thread);
  for (int tid = 0; tid < num_thread; ++tid) {
    for (int i = 0; i < 2; ++i) {
      buffers[tid][i].Alloc(batch_size);
      result_addr[tid][i].Alloc(batch_size);

      redwood::AttachStreamMem(tid, i, buffers[tid][i].u_leaf_idx);
      redwood::AttachStreamMem(tid, i, buffers[tid][i].u_qs);
      redwood::AttachStreamMem(tid, i, result_addr[tid][i].underlying_dat);
    }
  }
}

inline void Release() {
  for (int tid = 0; tid < stored_num_threads; ++tid) {
    for (int i = 0; i < 2; ++i) {
      buffers[tid][i].DeAlloc();
      result_addr[tid][i].DeAlloc();
    }
  }

  redwood::UsmFree(lnt_base_addr);
}

inline void ResetBuffer(const int tid, const int cur_stream) {
  buffers[tid][cur_stream].Reset();
}

_NODISCARD inline float* RequestResultAddr(const int tid, const int stream_id,
                                           const int executor_index) {
  return result_addr[tid][stream_id].GetAddrAt(executor_index);
}

inline void ReduceLeafNode(const int tid, const int stream_id, const Task& task,
                           const int node_idx) {
  buffers[tid][stream_id].Push(task, node_idx);
}

inline void DebugCpuReduction(const Buffer& buf, const dist::Euclidean functor,
                              const ResultBuffer& results) {
  const auto n = buf.Size();

  // i is batch id, = tid, = index in the buffer
  for (int i = 0; i < n; ++i) {
    const auto node_idx = buf.u_leaf_idx[i];
    const auto q = buf.u_qs[i];

    const auto node_addr = LntDataAddrAt(node_idx);
    const auto addr = reinterpret_cast<KnnSet<float, 1>*>(results.GetAddrAt(i));

    for (int j = 0; j < stored_max_leaf_size; ++j) {
      const float dist = functor(node_addr[j], q);
      addr->Insert(dist);
    }
  }
}

inline void LaunchAsyncWorkQueue(const int tid, const int stream_id) {
  const auto num_active = buffers[tid][stream_id].Size();

  if constexpr (kDebugMod) {
    std::cout << "rdc::LaunchAsyncWorkQueue "
              << "tid: " << tid << ", stream: " << stream_id << ", "
              << buffers[tid][stream_id].Size() << " actives." << std::endl;
    // 128? 256?
  }

  dist::Euclidean functor{};
  redwood::NearestNeighborKernel(
      tid, stream_id, lnt_base_addr, stored_max_leaf_size,
      buffers[tid][stream_id].u_qs, buffers[tid][stream_id].u_leaf_idx,
      num_active, result_addr[tid][stream_id].underlying_dat, functor);
}
}  // namespace rdc
