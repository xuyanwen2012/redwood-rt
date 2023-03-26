#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "../cxxopts.hpp"
#include "../nn/AppParams.hpp"
#include "../nn/DistanceMetrics.hpp"
#include "../nn/KDTree.hpp"
#include "Executor.hpp"
#include "GlobalVars.hpp"
#include "ReducerHandler.hpp"
#include "Redwood.hpp"

_NODISCARD Point4F RandPoint() {
  return {
      MyRand(0, 1024),
      MyRand(0, 1024),
      MyRand(0, 1024),
      MyRand(0, 1024),
  };
}

int main(int argc, char** argv) {
  cxxopts::Options options("Nearest Neighbor (NN)",
                           "Redwood NN demo implementation");
  options.add_options()("f,file", "File name", cxxopts::value<std::string>())(
      "q,query", "Num to Query", cxxopts::value<int>()->default_value("16384"))(
      "p,thread", "Num Thread", cxxopts::value<int>()->default_value("1"))(
      "l,leaf", "Leaf node size", cxxopts::value<int>()->default_value("32"))(
      "b,batch_size", "Batch Size",
      cxxopts::value<int>()->default_value("1024"))(
      "c,cpu", "Enable Cpu Baseline",
      cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

  options.parse_positional({"file", "query"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (!result.count("file")) {
    std::cerr << "requires an input file (\"data/input_nn_1m_4f.dat\")\n";
    std::cout << options.help() << std::endl;
    exit(EXIT_FAILURE);
  }

  const auto data_file = result["file"].as<std::string>();
  app_params.max_leaf_size = result["leaf"].as<int>();
  app_params.m = result["query"].as<int>();
  app_params.batch_size = result["batch_size"].as<int>();
  app_params.num_threads = result["thread"].as<int>();
  app_params.cpu = result["cpu"].as<bool>();
  std::cout << app_params << std::endl;

  // Load/Prepare Data
  const auto in_data = load_data_from_file<Point4F>(data_file);
  const auto n = in_data.size();

  std::queue<Task> q_data;
  for (int i = 0; i < app_params.m; ++i) q_data.emplace(i, RandPoint());

  // Build tree
  const kdt::KdtParams params{app_params.max_leaf_size};
  tree_ref = std::make_shared<kdt::KdTree>(params, in_data.data(), n);

  const auto num_leaf_nodes = tree_ref->GetStats().num_leaf_nodes;
  auto lnt_addr = rdc::AllocateLnt(num_leaf_nodes, app_params.max_leaf_size);
  tree_ref->LoadPayload(lnt_addr);

  // Debug Settings
  final_results1.resize(app_params.m);
  final_results2.resize(app_params.m);

  // Init
  rdc::Init(app_params.batch_size);

  //if (app_params.cpu) {
    TimeTask("CPU Traversal", [&] {
      // Pure CPU traverse
      Executor cpu_exe{0, 0, 0};

      while (!q_data.empty()) {
        cpu_exe.SetQuery(q_data.front());
        cpu_exe.CpuTraverse();
        q_data.pop();

        // cpu_exe.result_set->DebugPrint();
      }
    });
  //} else {
    // Use Redwood
    constexpr auto num_streams = 2;

    constexpr auto tid = 0;

    std::vector<Executor> exes[num_streams];
    for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
      exes[stream_id].reserve(app_params.batch_size);
      for (int i = 0; i < app_params.batch_size; ++i) {
        exes[stream_id].emplace_back(tid, stream_id, i);
      }
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    auto cur_stream = 0;
    while (!q_data.empty()) {
      for (auto it = exes[cur_stream].begin(); it != exes[cur_stream].end();) {
        if (it->Finished()) {
          if (!q_data.empty()) {
            const auto q = q_data.front();
            q_data.pop();

            it->SetQuery(q);
            it->StartQuery();
          }

          ++it;
        } else {
          it->Resume();
          if (it->Finished()) {
            // Do not increment , let the same executor (it) take another task
          } else {
            ++it;
          }
        }
      }

      rdc::LaunchAsyncWorkQueue(cur_stream);

      // switch to next
      cur_stream = (cur_stream + 1) % num_streams;

      redwood::DeviceStreamSynchronize(cur_stream);
      rdc::ResetBuffer(tid, cur_stream);
    }

    redwood::DeviceSynchronize();

    const auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_streams; ++i) {
      for (auto& ex : exes[cur_stream]) {
        ex.CpuTraverse2();
      }
      cur_stream = (cur_stream + 1) % num_streams;
    }

    const auto t2 = std::chrono::high_resolution_clock::now();
    const auto time_span1 =
        std::chrono::duration_cast<std::chrono::duration<float>>(t1 - t0);
    std::cout << "Finished "
              << "! Time took: " << time_span1.count() << "s. " << std::endl;
  //}

  PrintFinalResult(final_results1, 32);
  PrintFinalResult(final_results2, 32);

  rdc::Release();
  return EXIT_SUCCESS;
}
