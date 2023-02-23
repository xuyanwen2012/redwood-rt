#pragma once

#include "Point.hpp"

namespace rdc {

// struct ReducerHandler {
//   void Init(const int batch_size = 1024);
//   void Release();
// };

inline void InitReducers();
inline void ReleaseReducers();
inline void SetQuery(const int tid, const int stream_id, const Point4F q);
inline void ReduceLeafNode(const int tid, const int stream_id,
                           const int node_idx);
inline void ReduceBranchNode(const int tid, const int stream_id,
                             const Point4F data);
inline void ClearBuffer(const int tid, const int stream_id);

inline int NextStream(const int stream_id);

// // Mostly for KNN
// template <typename T>
// _NODISCARD T* GetResultAddr(const int tid, const int stream_id) {
//   return rhs[tid].UsmResultAddr(stream_id);
// }

// // Mostly for BH/NN
// template <typename T>
// _NODISCARD T GetResultValueUnchecked(const int tid, const int stream_id) {
//   return *GetResultAddr<T>(tid, stream_id);
// }

// inline void LuanchKernelAsync(const int tid, const int stream_id) {
//   // TODO: Need to select User's kernel
//   redwood::ComputeOneBatchAsync(
//       rhs[tid].UsmBuffer(stream_id).Data(), /* Buffered data to process */
//       static_cast<int>(rhs[tid].UsmBuffer(stream_id).Size()), /* / */
//       rhs[tid].UsmResultAddr(stream_id),                      /* Return Addr
//       */ rdc::LntDataAddr(),                                     /* Shared
//       data */ nullptr,                        /* Ignore for now */
//       rhs[tid].QueryPoint(stream_id), /* Single data */
//       stream_id);
// }

}  // namespace rdc