#pragma once

#include "Constants.hpp"
#include "Point.hpp"

namespace rdc {

constexpr auto kNumStreams = redwood::kNumStreams;

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
    return (kNumStreams - 1) - stream_id;
  }

  template <typename... Args>
  static ResultT* GetResultAddr(Args&&... args) {
    return Derived::GetResultAddr(std::forward<Args>(args)...);
  }
};

}  // namespace rdc