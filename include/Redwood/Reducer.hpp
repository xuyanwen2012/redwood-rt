#pragma once

#include <utility>

#include "Constants.hpp"

namespace rdc {

// Algorithms should implement
// Common API interface for Reducer
//
template <typename Derived, typename DataT, typename ResultT>
struct ReducerBase {
  static void InitReducers() { Derived::InitReducers(); }
  static void ReleaseReducers() { Derived::ReleaseReducers(); }

  template <typename... Args>
  static void SetQuery(Args&&... args) {
    Derived::SetQuery(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void ReduceLeafNode(Args&&... args) {
    Derived::ReduceLeafNode(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void ReduceBranchNode(Args&&... args) {
    Derived::ReduceBranchNode(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static void ClearBuffer(Args&&... args) {
    Derived::ClearBuffer(std::forward<Args>(args)...);
  }

  static int NextStream(const int stream_id) {
    return (redwood::kNumStreams - 1) - stream_id;
  }

  template <typename... Args>
  static ResultT* GetResultAddr(Args&&... args) {
    return Derived::GetResultAddr(std::forward<Args>(args)...);
  }
};

}  // namespace rdc