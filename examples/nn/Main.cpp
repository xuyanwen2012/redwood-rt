#include <iostream>
#include <vector>

#include "../KDTree.hpp"
#include "../LoadFile.hpp"
#include "ExecutorManager.hpp"
#include "HostKernelFunc.hpp"
#include "PointCloud.hpp"
#include "Redwood.hpp"
#include "UsmAlloc.hpp"

float CpuNaiveQuery(const Point4F* in_data, const Point4F q, const unsigned n) {
  constexpr auto kernel_func = MyFunctorHost();

  std::vector<float> dists(n);
  std::transform(in_data, in_data + n, dists.begin(),
                 [&](const auto& p) { return kernel_func(p, q); });

  return *std::min_element(dists.begin(), dists.end());
}

int main() {
  const auto [in_data, n] = LoadData<Point4F>("../../data/in_4f.dat");
  const auto [q_data, m] = LoadData<Point4F>("../../data/q_4f.dat");

  const auto leaf_size = 32;
  const auto num_batches = 1024;

  std::cout << "Simulation Parameters" << '\n';
  std::cout << "\tN: " << n << '\n';
  std::cout << "\tM: " << m << '\n';
  std::cout << "\tLeaf Size: " << leaf_size << '\n';
  std::cout << "\tNum Batches: " << num_batches << '\n';
  std::cout << std::endl;

  redwood::InitReducer(1, leaf_size, 1024, 1);

  const auto tid = 0;

  redwood::SetQueryPoints(tid, q_data, m);

  // Build tree
  std::cout << "Building KD Tree... " << '\n';

  const kdt::KdtParams params{leaf_size};
  // kdt::KdTree kdt(params, in_data, n);
  auto kdt = std::make_shared<kdt::KdTree>(params, in_data, n);

  // Load leaf node data into USM
  const auto num_leaf_nodes = kdt->GetStats().num_leaf_nodes;
  redwood::UsmVector<Point4F> leaf_node_table(num_leaf_nodes * leaf_size);
  kdt->LoadPayload(leaf_node_table.data());
  redwood::SetNodeTables(leaf_node_table.data(), num_leaf_nodes);

  // Redwood
  std::vector<int> q_idx(m);
  std::iota(q_idx.begin(), q_idx.end(), 0);

  ExecutorManager manager(kdt, q_data, q_idx.data(), m, num_batches, tid);

  manager.StartTraversals();

  // Display Results
  for (int i = 0; i < 5; ++i) {
    float* rst;
    redwood::GetReductionResult(tid, i, &rst);
    std::cout << "Query " << i << ":\n"
              << "\tQuery point " << q_data[i] << '\n'
              << "\tresult:      \t" << *rst << '\n';

    if constexpr (constexpr auto show_ground_truth = true) {
      std::cout << "\tground_truth: \t" << CpuNaiveQuery(in_data, q_data[i], n)
                << '\n';
    }
    std::cout << std::endl;
  }

  redwood::EndReducer();

  return 0;
}