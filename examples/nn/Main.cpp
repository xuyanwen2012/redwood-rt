#include <iostream>
#include <vector>

#include "../KDTree.hpp"
#include "../LoadFile.hpp"
#include "PointCloud.hpp"
#include "Redwood.hpp"
#include "UsmAlloc.hpp"

int main() {
  const auto [in_data, n] = LoadData<Point4F>("../../data/in_4f.dat");
  const auto [q_data, m] = LoadData<Point4F>("../../data/q_4f.dat");

  const auto leaf_size = 32;

  std::cout << "Simulation Parameters" << '\n';
  std::cout << "\tN: " << n << '\n';
  std::cout << "\tM: " << m << '\n';
  std::cout << "\tLeaf Size: " << leaf_size << '\n';
  std::cout << std::endl;

  redwood::InitReducer(1, leaf_size, 1024, 1);

  const auto tid = 0;

  redwood::SetQueryPoints(tid, q_data, m);

  // Build tree
  std::cout << "Building KD Tree... " << '\n';

  const kdt::KdtParams params{leaf_size};
  kdt::KdTree kdt(params, in_data, n);

  const auto num_leaf_nodes = kdt.GetStats().num_leaf_nodes;
  redwood::UsmVector<Point4F> leaf_node_table(num_leaf_nodes * leaf_size);
  kdt.LoadPayload(leaf_node_table.data());
  redwood::SetNodeTables(leaf_node_table.data(), num_leaf_nodes);

  const int fake_indecies[6] = {5, 6, 1, 2, 3, 6};
  for (int query_idx = 0; query_idx < 6; ++query_idx) {
    redwood::ReduceLeafNode(tid, fake_indecies[query_idx], query_idx);
  }

  redwood::rt::ExecuteCurrentBufferAsync(tid, 6);

  const int fake_indecies2[6] = {1, 2, 3, 4, 5, 6};
  for (int query_idx = 0; query_idx < 6; ++query_idx) {
    redwood::ReduceLeafNode(tid, fake_indecies2[query_idx], query_idx);
  }

  redwood::rt::ExecuteCurrentBufferAsync(tid, 6);

  float result = 666.6f;
  redwood::GetReductionResult(tid, 0, &result);

  std::cout << "Result: " << result << std::endl;

  redwood::EndReducer();

  return 0;
}