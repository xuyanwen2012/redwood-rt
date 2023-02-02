#include <iostream>
#include <vector>

#include "LoadFile.hpp"
#include "Redwood.hpp"
#include "UsmAlloc.hpp"

int main() {
  const auto [in_data, n] = LoadData("../data/test.dat");
  const auto [q_data, m] = LoadData("../data/query.dat");

  const auto leaf_size = 32;

  redwood::InitReducer(1, leaf_size, 1024, 1);

  const auto tid = 0;

  redwood::SetQueryPoints(tid, q_data, m);

  // Build tree
  redwood::UsmVector<float> leaf_node_table;

  // I think you need to estimate num_leaf_nodes, or reprocess

  // Pretent we are building a tree
  const auto num_leaf_nodes = n / leaf_size;
  leaf_node_table.resize(num_leaf_nodes * leaf_size);

  for (auto i = 0u; i < n; ++i) {
    leaf_node_table[i] = in_data[i];
  }

  redwood::SetNodeTables(leaf_node_table.data(), num_leaf_nodes);

  const int fake_indecies[6] = {5, 6, 1, 2, 3, 6};
  for (int query_idx = 0; query_idx < 6; ++query_idx) {
    redwood::ReduceLeafNode(tid, fake_indecies[query_idx], query_idx);
  }

  redwood::rt::ExecuteCurrentBufferAsync(tid, 6);
  // redwood::rt::ExecuteBuffer(tid,0, 6);

  const int fake_indecies2[6] = {1, 2, 3, 4, 5, 6};
  for (int query_idx = 0; query_idx < 6; ++query_idx) {
    redwood::ReduceLeafNode(tid, fake_indecies2[query_idx], query_idx);
  }

  redwood::rt::ExecuteCurrentBufferAsync(tid, 6);

  const int fake_indecies3[6] = {6, 5, 4, 3, 2, 1};
  for (int query_idx = 0; query_idx < 6; ++query_idx) {
    redwood::ReduceLeafNode(tid, fake_indecies3[query_idx], query_idx);
  }

  redwood::rt::ExecuteCurrentBufferAsync(tid, 6);

  redwood::EndReducer();

  return 0;
}