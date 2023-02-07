#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "../KDTree.hpp"
// #include "../LoadFile.hpp"

// Redwood related
#include "PointCloud.hpp"
#include "Redwood.hpp"
#include "UsmAlloc.hpp"

float my_rand(float min = 0.0, float max = 1.0) {
  // 114514 and 233
  static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
  std::uniform_real_distribution<float> distribution(min, max);
  return distribution(generator);
}

int main() {
  // const auto [in_data, n] = LoadData<Point4F>("../../data/in_4f.dat");
  // const auto [q_data, m] = LoadData<Point4F>("../../data/q_4f.dat");

  const auto n = 1024 * 128;
  const auto m = 1024 * 32;

  auto in_data = static_cast<Point4F *>(malloc(n * sizeof(Point4F)));
  auto q_data = static_cast<Point4F *>(malloc(m * sizeof(Point4F)));

  static auto rand_point4f = []() {
    return Point4F{
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
        my_rand(0.0f, 1000.0f),
    };
  };

  std::generate_n(in_data, n, rand_point4f);
  std::generate_n(q_data, m, rand_point4f);

  const auto leaf_size = 32;  // TODO: argument
  const auto num_batches = 1024;
  const auto batch_size = 1024;

  std::cout << "Simulation Parameters" << '\n';
  std::cout << "\tN: " << n << '\n';
  std::cout << "\tM: " << m << '\n';
  std::cout << "\tLeaf Size: " << leaf_size << '\n';
  std::cout << "\tNum Batches: " << num_batches << '\n';
  std::cout << std::endl;

  redwood::InitReducer(1, leaf_size, 1024, batch_size);

  const auto tid = 0;

  redwood::SetQueryPoints(tid, q_data, m);

  // // Build tree
  // std::cout << "Building KD Tree... " << '\n';

  // const kdt::KdtParams params{leaf_size};
  // auto kdt = std::make_shared<kdt::KdTree>(params, in_data, n);

  // // Load leaf node data into USM
  // const auto num_leaf_nodes = kdt->GetStats().num_leaf_nodes;

  // redwood::UsmVector<Point4F> leaf_node_table(num_leaf_nodes * leaf_size);
  // kdt->LoadPayload(leaf_node_table.data());
  // redwood::SetNodeTables(leaf_node_table.data(), num_leaf_nodes);

  // // Redwood
  // std::vector<int> q_idx(m);
  // std::iota(q_idx.begin(), q_idx.end(), 0);

  // ExecutorManager manager(kdt, q_data, q_idx.data(), m, num_batches, tid);

  // manager.StartTraversals();

  // // Display Results
  // for (int i = 0; i < 5; ++i) {
  //   float *rst;
  //   redwood::GetReductionResult(tid, i, &rst);
  //   std::cout << "Query " << i << ":\n"
  //             << "\tQuery point " << q_data[i] << '\n'
  //             << "\tresult:      \t" << *rst << '\n';

  //   if constexpr (constexpr auto show_ground_truth = true) {
  //     std::cout << "\tground_truth: \t" << CpuNaiveQuery(in_data, q_data[i],
  //     n)
  //               << '\n';
  //   }
  //   std::cout << std::endl;
  // }

  redwood::EndReducer();

  return 0;
}