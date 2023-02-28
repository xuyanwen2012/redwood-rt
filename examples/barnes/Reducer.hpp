#pragma once

#include <array>

#include "../LeafNodeTable.hpp"
#include "Redwood.hpp"
#include "Redwood/Kernels.hpp"

namespace rdc {

template <typename DataT, typename ResultT>
struct BarnesHandler {
  using IndicesBuffer = redwood::UsmVector<int>;

 private:
  // Each 'Buffer' contains three piece of informations
  // 1) Leaf node indices collected by the traverser algorithm, stored in USM
  // 2) A region of memory (4 bytes) for writting result, stored in USM
  // 3) A single query point 'q', could be on host.
  //
  // And we need two 'buffer' for double buffering
  //
  std::array<IndicesBuffer, redwood::kNumStreams> usm_leaf_nodes;
  std::array<ResultT*, redwood::kNumStreams> usm_result;
  std::array<DataT, redwood::kNumStreams> h_query;

 public:
  void Init(const int batch_size = 1024) {
    for (int i = 0; i < redwood::kNumStreams; ++i) {
      usm_leaf_nodes[i].reserve(batch_size);

      constexpr auto result_size = 1;
      usm_result[i] = redwood::UsmMalloc<float>(result_size);

      if constexpr (kRedwoodBackend == redwood::Backends::kCuda) {
        // CUDA Only
        redwood::AttachStreamMem(i, usm_leaf_nodes[i].data());
        redwood::AttachStreamMem(i, usm_result[i]);
      }
    }
  }

  void Release() {
    for (int i = 0; i < redwood::kNumStreams; ++i) {
      redwood::UsmFree(usm_result[i]);

      // Mannuelly free
      IndicesBuffer tmp;
      usm_leaf_nodes[i].swap(tmp);
    }
  }

  _NODISCARD IndicesBuffer& UsmBuffer(const int stream_id) {
    return usm_leaf_nodes[stream_id];
  }

  _NODISCARD ResultT* UsmResultAddr(const int stream_id) {
    return usm_result[stream_id];
  }

  _NODISCARD const DataT QueryPoint(const int stream_id) const {
    return h_query[stream_id];
  }

  void SetQueryPoint(const int stream_id, const DataT q) {
    h_query[stream_id] = q;
  }
};

using MyBarnesHandler = BarnesHandler<Point4F, float>;

template <typename DataT, typename ResultT>
struct DoubleBufferReducer
    : ReducerBase<DoubleBufferReducer<DataT, ResultT>, DataT, ResultT> {
 private:
  static std::vector<MyBarnesHandler> rhs;
  static int num_threads;

 public:
  static void InitReducers() { num_threads = 1; }

  static void ReleaseReducers() {
    for (int i = 0; i < num_threads; ++i) rhs[i].Release();
  }

  static void SetQuery(const int tid, const int stream_id, const DataT q) {
    rhs[tid].SetQueryPoint(stream_id, q);
  }

  static void ReduceLeafNode(const int tid, const int stream_id,
                             const int node_idx) {
    rhs[tid].UsmBuffer(stream_id).push_back(node_idx);
  }

  static void ReduceBranchNode(const int tid, const int stream_id,
                               const DataT data) {
    // TODO: Do reduction on CPU for now
  }

  static void ClearBuffer(const int tid, const int stream_id) {
    rhs[tid].UsmBuffer(stream_id).clear();
  }

  static ResultT* GetResultAddr(const int tid, const int stream_id) {
    return rhs[tid].UsmResultAddr(stream_id);
  }

  // TODO: Make sure in future version, we can launch User generated Kernel
  static void LuanchKernelAsync(const int tid, const int stream_id) {
    // redwood::ComputeOneBatchAsync(
    //     rhs[tid].UsmBuffer(stream_id).data(), /* Buffered data to process */
    //     static_cast<int>(rhs[tid].UsmBuffer(stream_id).size()),
    //     rhs[tid].UsmResultAddr(stream_id), /* Return Addr */
    //     rdc::LntDataAddr(),                /* Shareddata */
    //     nullptr,                           /* Ignore for now */
    //     rhs[tid].QueryPoint(stream_id),    /* Single data */
    //     stream_id);
    redwood::ComputeOneBatchAsync(
        // rhs[tid].UsmBuffer(stream_id).data(), /* Buffered data to process */
        // static_cast<int>(rhs[tid].UsmBuffer(stream_id).size()),

        // NodeDataBaseAddr(node_idx),
        // HowMany,

        rhs[tid].UsmResultAddr(stream_id), /* Return Addr */
        rdc::LntDataAddr(),                /* Shareddata */
        nullptr,                           /* Ignore for now */
        rhs[tid].QueryPoint(stream_id),    /* Single data */
        stream_id);
  }
};

// template <typename DataT, typename ResultT>
// struct DoubleBufferReducer
//     : ReducerBase<DoubleBufferReducer<DataT, ResultT>, DataT, ResultT> {
//  private:
//   static std::array<MyBarnesHandler, kNumThreads> rhs;

//  public:
//   static void InitReducers() {}

//   static void ReleaseReducers() {
//     for (int i = 0; i < kNumThreads; ++i) rhs[i].Release();
//   }

//   static void SetQuery(const int tid, const int stream_id, const DataT q) {
//     rhs[tid].SetQueryPoint(stream_id, q);
//   }

//   static void ReduceLeafNode(const int tid, const int stream_id,
//                              const int node_idx) {
//     rhs[tid].UsmBuffer(stream_id).PushLeaf(node_idx);
//   }

//   static void ReduceBranchNode(const int tid, const int stream_id,
//                                const DataT data) {}

//   static void ClearBuffer(const int tid, const int stream_id) {
//     rhs[tid].UsmBuffer(stream_id).Clear();
//   }

//   static ResultT* GetResultAddr(const int tid, const int stream_id) {
//     return rhs[tid].UsmResultAddr(stream_id);
//   }

//   static void LuanchKernelAsync(const int tid, const int stream_id) {
//     redwood::ComputeOneBatchAsync(
//         rhs[tid].UsmBuffer(stream_id).Data(), /* Buffered data to process */
//         static_cast<int>(rhs[tid].UsmBuffer(stream_id).Size()), /* / */
//         rhs[tid].UsmResultAddr(stream_id), /* Return Addr */
//         rdc::LntDataAddr(),                /* Shareddata */
//         nullptr,                           /* Ignore for now */
//         rhs[tid].QueryPoint(stream_id),    /* Single data */
//         stream_id);
//   }
// };

}  // namespace rdc