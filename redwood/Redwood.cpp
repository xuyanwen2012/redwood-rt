#include "../include/Redwood.hpp"

#include <array>
#include <iostream>

#include "../include/UsmAlloc.hpp"
#include "Kernel.hpp"
#include "SharedData.hpp"

namespace redwood {

// ------------------- Constants -------------------

constexpr auto kNumStreams = 2;
int stored_leaf_size;
int stored_num_batches;
int stored_num_threads;
// int stored_num_leaf_nodes

template <typename T>
struct NnBuffer {
  void Allocate(const int num_batch) {
    query_point.reserve(num_batch);
    query_idx.reserve(num_batch);
    leaf_idx.reserve(num_batch);
  }

  size_t Size() const { return leaf_idx.size(); }

  void Clear() {
    // TODO: no need to clear every time, just overwrite the value
    query_point.clear();
    query_idx.clear();
    leaf_idx.clear();
  }

  void Push(const T& q, const int q_idx, const int leaf_id) {
    query_point.push_back(q);
    query_idx.push_back(q_idx);
    leaf_idx.push_back(leaf_id);

    std::cout << q << " Pushed. " << leaf_id << "( " << Size() << "/"
              << leaf_idx.capacity() << ")" << std::endl;
  }

  UsmVector<T> query_point;
  UsmVector<int> query_idx;
  UsmVector<int> leaf_idx;
};

template <typename T>
struct NnResult {
  NnResult(const int num_query) : results(num_query) {
    std::fill(results.begin(), results.end(), std::numeric_limits<T>::max());
  }

  UsmVector<T> results;
};

struct ReducerHandler {
  void Init(const int batch_num) {
    for (int i = 0; i < kNumStreams; ++i) {
      nn_buffers[i].Allocate(batch_num);

      internal::AttachStreamMem(i, nn_buffers[i].query_point.data());
      internal::AttachStreamMem(i, nn_buffers[i].query_idx.data());
      internal::AttachStreamMem(i, nn_buffers[i].leaf_idx.data());
    }
  };

  std::array<NnBuffer<float>, kNumStreams> nn_buffers;
  std::vector<NnResult<float>> nn_results;

  NnBuffer<float>& CurrentBuffer() { return nn_buffers[cur_collecting]; }
  NnResult<float>& CurrentResult() { return nn_results[cur_collecting]; }

  int cur_collecting = 0;
};

// ------------------- Global Shared  -------------------

SharedData sdata;
ReducerHandler* rhs;

// ------------------- Public APIs  -------------------

void InitReducer(const int num_threads, const int leaf_size,
                 const int batch_num, const int batch_size) {
  stored_leaf_size = leaf_size;
  stored_num_threads = num_threads;
  stored_num_batches = batch_num;
  internal::DeviceWarmUp();

  rhs = new ReducerHandler[num_threads];
  for (int i = 0; i < num_threads; ++i) {
    rhs[i].Init(batch_num);
  }
}

void SetQueryPoints(const int tid, const void* query_points,
                    const int num_query) {
  sdata.host_query_point_ref = static_cast<const float*>(query_points);

  rhs[tid].nn_results.reserve(kNumStreams);
  rhs[tid].nn_results.emplace_back(num_query);
  rhs[tid].nn_results.emplace_back(num_query);

  internal::AttachStreamMem(0, rhs[tid].nn_results[0].results.data());
  internal::AttachStreamMem(1, rhs[tid].nn_results[1].results.data());
}

void SetNodeTables(const void* usm_leaf_node_table, const int num_leaf_nodes) {
  sdata.usm_leaf_node_table = static_cast<const float*>(usm_leaf_node_table);

  internal::RegisterLeafNodeTable(usm_leaf_node_table, num_leaf_nodes);
}

void ReduceLeafNode(const int tid, const int node_idx, const int query_idx) {
  rhs[tid].CurrentBuffer().Push(sdata.host_query_point_ref[query_idx],
                                query_idx, node_idx);
}

void EndReducer() { delete[] rhs; }

// ------------------- Developer APIs  -------------------

void rt::ExecuteCurrentBufferAsync(int tid, int num_batch_collected) {
  const auto& cb = rhs[tid].CurrentBuffer();
  const auto current_stream = rhs[tid].cur_collecting;

  std::cout << "rt::ExecuteCurrentBufferAsync() " << current_stream
            << std::endl;

  internal::ProcessNnBuffer(cb.query_point.data(), cb.query_idx.data(),
                            cb.leaf_idx.data(), nullptr,
                            rhs[tid].CurrentResult().results.data(),
                            num_batch_collected, current_stream);

  const auto next_stream = (kNumStreams - 1) - current_stream;
  internal::DeviceStreamSynchronize(next_stream);

  rhs[tid].nn_buffers[next_stream].Clear();
  rhs[tid].cur_collecting = next_stream;
}

void rt::ExecuteBuffer(int tid, int stream_id, int num_batch_collected) {
  const auto& cb = rhs[tid].CurrentBuffer();

  internal::ProcessNnBuffer(
      cb.query_point.data(), cb.query_idx.data(), cb.leaf_idx.data(), nullptr,
      rhs[tid].CurrentResult().results.data(), num_batch_collected, stream_id);

  internal::DeviceSynchronize();

  for (int i = 0; i < 6; ++i)
    std::cout << "Result: " << rhs[tid].CurrentResult().results[i] << std::endl;
  //   Size()
}

}  // namespace redwood