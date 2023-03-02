#pragma once

#include <array>

#include "../LeafNodeTable.hpp"
#include "Redwood.hpp"
#include "Redwood/Duet/DuetAPI.hpp"
#include "Redwood/Kernels.hpp"

namespace rdc {

// to be parameterized
// constexpr auto kNumThreads = 1;
constexpr auto kMaxLeafSize = 64;

template <typename DataT, typename ResultT>
struct DuetBarnesReducer
    : ReducerBase<DuetBarnesReducer<DataT, ResultT>, DataT, ResultT> {
  static void InitReducers() {}

  static void ReleaseReducers() {}

  static void SetQuery(const int tid, const int stream_id, const DataT& q) {
    duet::Start(tid, q);
  }

  static void ReduceLeafNode(const int tid, const int stream_id,
                             const int node_idx) {
    auto addr = LntDataAddrAt(node_idx);
    duet::PushLeaf32(tid, addr);
  }

  static void ReduceBranchNode(const int tid, const int stream_id,
                               const DataT data) {}

  static void ClearBuffer(const int tid, const int stream_id) {}

  static ResultT* GetResultAddr(const int tid, const int stream_id) {}

  static void LuanchKernelAsync(const int tid, const int stream_id) {}
};

}  // namespace rdc
