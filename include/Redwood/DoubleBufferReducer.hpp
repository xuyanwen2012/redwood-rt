#pragma once

#include <array>

#include "Constants.hpp"
#include "Point.hpp"

namespace rdc {

// struct ReducerHandler {
//   void Init(const int batch_size = 1024);
//   void Release();
// };

// These are Common apis,

constexpr auto kNumStreams = redwood::kNumStreams;

// 0) Init
// 1) Release
// 2) OnSetQuery
// 3) OnReduceLeaf
// 4) OnReduceBranch
// 5) OnBufferSwitch
// 6) GetResultAddr
// 7) LuanchKernelAsync

struct Handler {};

template <typename HandlerT, int kThreads = 1>
struct Reducer {
  using DataT = typename HandlerT::DataT;
  using ResultT = typename HandlerT::ResultT;

  static std::array<Handler, kThreads> rhs;

  static void InitReducers() {
    for (int i = 0; i < kThreads; ++i) rhs[i].Init();
  }

  static void ReleaseReducers() {
    for (int i = 0; i < kThreads; ++i) rhs[i].Release();
  }

  static void SetQuery(const int tid, const int stream_id, const DataT q) {
    rhs[tid].OnSetQuery(stream_id, q);
  }

  static void ReduceLeafNode(const int tid, const int stream_id,
                             const int node_idx) {
    // rhs[tid].UsmBuffer(stream_id).PushLeaf(node_idx);
  }

  static void ReduceBranchNode(const int tid, const int stream_id,
                               const DataT data) {}

  static void OnBufferSwitch(const int tid) {
    // rhs[tid].UsmBuffer(stream_id).Clear();
  }

  static ResultT* GetResultAddr(const int tid, const int stream_id) {
    return rhs[tid].UsmResultAddr(stream_id);
  }

  static void LuanchKernelAsync(const int tid, const int stream_id) {
    // TODO: Make this Function Pointer
  }
};

// template <typename DataT, typename ResultT>
// struct ReducerBase {
//   static void InitReducers() { Derived::InitReducers(); }
//   static void ReleaseReducers() { Derived::ReleaseReducers(); }

//   static void SetQuery(Args&&... args) {
//   }

//   static void ReduceLeafNode(Args&&... args) {
//   }

//   static void ReduceBranchNode(Args&&... args) {
//   }

//   static void ClearBuffer(Args&&... args) {
//   }

//   static int NextStream(const int stream_id) {
//     return (kNumStreams - 1) - stream_id;
//   }

//   static ResultT* GetResultAddr(Args&&... args) {
//   }
// };

}  // namespace rdc