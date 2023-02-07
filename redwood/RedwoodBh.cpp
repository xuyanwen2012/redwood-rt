#include <array>
#include <iostream>

#include "../include/PointCloud.hpp"
#include "../include/Redwood.hpp"
#include "../include/UsmAlloc.hpp"
#include "BhBuffer.hpp"
#include "Kernel.hpp"

namespace redwood {

// ------------------- Constants -------------------

constexpr auto kNumStreams = 2;
int stored_leaf_size;
int stored_num_batches;
int stored_batch_size;
int stored_num_threads;

template <typename DataT, typename QueryT, typename ResultT>
struct ReducerHandler {
  using MyBhBuffer = BhBuffer<DataT, QueryT, ResultT>;

  void Init(const int batch_num) {
    for (int i = 0; i < kNumStreams; ++i) {
      bh_buffers[i].Allocate(batch_num);

      internal::AttachStreamMem(i, bh_buffers[i].leaf_nodes.data());
      internal::AttachStreamMem(i, bh_buffers[i].branch_data.data());
    }
  };

  std::array<MyBhBuffer, kNumStreams> bh_buffers;
  std::array<UsmVector<ResultT>, kNumStreams> bh_results;

  MyBhBuffer& CurrentBuffer() { return bh_buffers[cur_collecting]; }
  ResultT* CurrentResultData() { return bh_results[cur_collecting].data(); }

  int cur_collecting = 0;
};

// ------------------- Global Shared  -------------------

const Point3F* host_query_point_ref;
const Point4F* usm_leaf_node_table_ref;

ReducerHandler<Point4F, Point3F, Point3F>* rhs;

// ------------------- Public APIs  -------------------

void InitReducer(const int num_threads, const int leaf_size,
                 const int batch_num, const int batch_size) {
  stored_leaf_size = leaf_size;
  stored_num_threads = num_threads;
  stored_num_batches = batch_num;

  internal::BackendInitialization();

  rhs = new ReducerHandler<Point4F, Point3F, Point3F>[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    rhs[i].Init(batch_num);
  }
}

void SetQueryPoints(const int tid, const void* query_points,
                    const int num_query) {
  host_query_point_ref = static_cast<const Point3F*>(query_points);

  rhs[tid].bh_results[0].resize(num_query);
  rhs[tid].bh_results[1].resize(num_query);

  internal::AttachStreamMem(0, rhs[tid].bh_results[0].data());
  internal::AttachStreamMem(1, rhs[tid].bh_results[1].data());
}

void SetNodeTables(const void* usm_leaf_node_table, const int num_leaf_nodes) {
  usm_leaf_node_table_ref = static_cast<const Point4F*>(usm_leaf_node_table);
  // internal::RegisterLeafNodeTable(usm_leaf_node_table, num_leaf_nodes);
}

void StartQuery(const int tid, const int query_idx) {
  rhs[tid].CurrentBuffer().SetTask(host_query_point_ref[query_idx], query_idx);
}

void ReduceLeafNode(const int tid, const int node_idx, const int query_idx) {
  rhs[tid].CurrentBuffer().PushLeaf(node_idx);
}

void ReduceBranchNode(int tid, const void* node_element, int query_idx) {
  rhs[tid].CurrentBuffer().PushBranch(
      *static_cast<const Point4F*>(node_element));
}

void GetReductionResult(const int tid, const int query_idx, void* result) {
  auto addr = static_cast<Point3F**>(result);
  *addr = &rhs[tid].CurrentResultData()[query_idx];
}

void EndReducer() {
  // const auto tid = 0;
  // const auto current_stream = rhs[tid].cur_collecting;
  // const auto next_stream = (kNumStreams - 1) - current_stream;
  // internal::DeviceStreamSynchronize(next_stream);

  // // Only for Sycl
  // const auto qx = rhs[tid].bh_buffers[next_stream].my_q_idx;
  // internal::OnBhBufferFinish(rhs[tid].bh_results[next_stream].data() + qx,
  //                            next_stream);

  delete[] rhs;
}

// ------------------- Developer APIs  -------------------

void rt::ExecuteCurrentBufferAsync(int tid, int num_batch_collected) {
  const auto& cb = rhs[tid].CurrentBuffer();
  const auto current_stream = rhs[tid].cur_collecting;

  // The current implementation process on query only
  internal::ProcessBhBuffer(
      cb.my_query, usm_leaf_node_table_ref, cb.LeafNodeData(),
      cb.NumLeafsCollected(), cb.BranchNodeData(), cb.NumBranchCollected(),
      rhs[tid].CurrentResultData(), stored_leaf_size, current_stream);

  std::cout << "DEBUG: " << *rhs[tid].CurrentResultData() << std::endl;

  const auto next_stream = (kNumStreams - 1) - current_stream;
  internal::DeviceStreamSynchronize(next_stream);

  // // // Only for Sycl
  // internal::OnBhBufferFinish(rhs[tid].bh_results[next_stream].data(),
  //                            next_stream);

  rhs[tid].bh_buffers[next_stream].Clear();
  rhs[tid].cur_collecting = next_stream;
}

}  // namespace redwood