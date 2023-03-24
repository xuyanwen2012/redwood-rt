#pragma once

#include <array>

#include "../Utils.hpp"
#include "AppParams.hpp"
#include "DistanceMetrics.hpp"
#include "KnnSet.hpp"
#include "Redwood/Point.hpp"

using Task = std::pair<int, Point4F>;

#include "Redwood/Usm.hpp"

namespace rdc {

inline redwood::UsmVector<Point4F> lnt;

_NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
  return lnt.data() + node_idx * app_params.max_leaf_size;
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

std::array<Buffer, 2> buffers;
std::array<ResultBuffer, 2> result_addr;

void Init(const int batch_size) {
  for (int i = 0; i < 2; ++i) {
    buffers[i].Alloc(batch_size);
    result_addr[i].Alloc(batch_size);
  }
}

void Release() {
  for (int i = 0; i < 2; ++i) {
    buffers[i].DeAlloc();
    result_addr[i].DeAlloc();
  }
}

void ResetBuffer(const int tid, const int cur_stream) {
  buffers[cur_stream].Reset();
}

_NODISCARD float* RequestResultAddr(const int stream_id,
                                    const int executor_index) {
  return result_addr[stream_id].GetAddrAt(executor_index);
}

void ReduceLeafNode(const int stream_id, const Task& task, const int node_idx) {
  buffers[stream_id].Push(task, node_idx);
}

void DebugCpuReduction(const Buffer& buf, const dist::Euclidean functor,
                       const ResultBuffer& results) {
  const auto n = buf.Size();

  // i is batch id, = tid, = index in the buffer
  for (int i = 0; i < n; ++i) {
    const auto node_idx = buf.u_leaf_idx[i];
    const auto q = buf.u_qs[i];

    const auto node_addr = LntDataAddrAt(node_idx);
    const auto addr = reinterpret_cast<KnnSet*>(results.GetAddrAt(i));

    for (int j = 0; j < app_params.max_leaf_size; ++j)

    {
      const float dist = functor(node_addr[j], q);
      addr->Insert(dist);
    }
  }
}

void LaunchAsyncWorkQueue(const int stream_id) {
  std::cout << "rdc::LaunchAsyncWorkQueue " << stream_id << ", "
            << buffers[stream_id].Size() << " actives." << std::endl;

  DebugCpuReduction(buffers[stream_id], dist::Euclidean(),
                    result_addr[stream_id]);

  // You may reset the buffer?
}
}  // namespace rdc