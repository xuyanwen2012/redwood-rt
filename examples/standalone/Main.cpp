#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

#include "../Utils.hpp"
#include "../cxxopts.hpp"
#include "AppParams.hpp"
#include "DistanceMetrics.hpp"
#include "Executor.hpp"
#include "GlobalVars.hpp"
#include "KDTree.hpp"
#include "LoadFile.hpp"
#include "ReducerHandler.hpp"
#include "Redwood/Redwood.hpp"

Point4F RandPoint() {
  return {
      MyRand(0, 1024),
      MyRand(0, 1024),
      MyRand(0, 1024),
      MyRand(0, 1024),
  };
}

void PrintLeafNodeVisited(const std::vector<std::vector<int>>& d) {
  for (auto i = 0u; i < d.size(); ++i) {
    std::cout << "Query " << i << ": [";
    for (const auto& elem : d[i]) {
      std::cout << elem;
      if (elem != d[i].back()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

void PrintFinalResult(const std::vector<float>& d) {
  for (auto i = 0u; i < d.size(); ++i)
    std::cout << "Query " << i << ": " << d[i] << '\n';
  std::cout << std::endl;
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

  // Config
  const auto data_file = result["file"].as<std::string>();
  app_params.max_leaf_size = result["leaf"].as<int>();
  app_params.m = result["query"].as<int>();
  app_params.batch_size = result["batch_size"].as<int>();
  app_params.num_threads = result["thread"].as<int>();
  app_params.cpu = result["cpu"].as<bool>();

  // Loaded
  std::cout << app_params << std::endl;

  const auto in_data = load_data_from_file<Point4F>(data_file);
  const auto n = in_data.size();

  // Debug
  leaf_node_visited1.resize(app_params.m);
  leaf_node_visited2.resize(app_params.m);
  final_results1.resize(app_params.m);
  final_results2.resize(app_params.m);

  // Query (x2)
  std::queue<Task> q1_data;
  for (int i = 0; i < app_params.m; ++i) q1_data.emplace(i, RandPoint());
  std::queue q2_data(q1_data);

  // Init
  rdc::Init(app_params.batch_size);

  // Build tree
  {
    const kdt::KdtParams params{app_params.max_leaf_size};
    tree_ref = std::make_shared<kdt::KdTree>(params, in_data.data(), n);

    const auto num_leaf_nodes = tree_ref->GetStats().num_leaf_nodes;

    // Basically alloc
    auto lnt_addr = rdc::AllocateLnt(num_leaf_nodes, app_params.max_leaf_size);
    tree_ref->LoadPayload(lnt_addr);
  }

  // Pure CPU traverse
  {
    Executor cpu_exe{0, 0, 0};

    while (!q1_data.empty()) {
      cpu_exe.SetQuery(q1_data.front());
      cpu_exe.CpuTraverse();
      q1_data.pop();
    }

    // PrintLeafNodeVisited(leaf_node_visited1);
    if constexpr (kDebugMod) {
      PrintFinalResult(final_results1);
      std::cout << std::endl;
    }
  }

  // Traverser traverse (double buffer)
  {
    constexpr auto tid = 0;

    constexpr auto num_streams = 1;

    std::vector<Executor> exes[num_streams];
    for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
      exes[stream_id].reserve(app_params.batch_size);
      for (int i = 0; i < app_params.batch_size; ++i) {
        exes[stream_id].emplace_back(tid, stream_id, i);
      }
    }

    auto cur_stream = 0;
    while (!q2_data.empty()) {
      for (auto it = exes[cur_stream].begin(); it != exes[cur_stream].end();) {
        if (it->Finished()) {
          // final_results2[it->my_task_.first] = it->result_set->WorstDist();

          if (!q2_data.empty()) {
            const auto q = q2_data.front();
            q2_data.pop();

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

    for (auto& ex : exes[cur_stream]) {
      ex.ResumeNonStop();
    }

    //// Still some remaining
    // int num_incomplete[num_streams];
    // bool need_work;
    // do
    //{
    //	num_incomplete[cur_stream] = 0;
    //	for (auto& ex : exes[cur_stream])
    //	{
    //		if (!ex.Finished())
    //		{
    //			ex.Resume();
    //			++num_incomplete[cur_stream];
    //		}
    //		else
    //		{
    //			// final_results2[ex.my_task_.first] =
    // ex.result_set->WorstDist();
    //		}
    //	}

    //	rdc::LaunchAsyncWorkQueue(cur_stream);

    //	const auto next = (cur_stream + 1) % num_streams;
    //	cur_stream = next;
    //	redwood::DeviceStreamSynchronize(cur_stream);

    //	rdc::buffers[cur_stream].Reset();

    //	// Both stream must complete
    //	need_work = false;
    //	for (const int i : num_incomplete) need_work |= i > 0;
    //}
    // while (need_work);

    // PrintLeafNodeVisited(leaf_node_visited2);
    if constexpr (kDebugMod) {
      PrintFinalResult(final_results2);
    }
  }

  if constexpr (kDebugMod) {
    for (std::size_t i = 0; i < app_params.m; ++i) {
      const auto& inner1 = leaf_node_visited1[i];
      const auto& inner2 = leaf_node_visited2[i];

      std::cout << "Mismatched values in vector " << i << ":\n";

      for (std::size_t j = 0; j < inner1.size(); ++j) {
        if (j >= inner2.size() || inner1[j] != inner2[j]) {
          std::cout << inner1[j] << '\n';
        }
      }

      for (std::size_t j = inner1.size(); j < inner2.size(); ++j) {
        std::cout << inner2[j] << '\n';
      }

      std::cout << '\n';
    }
  }

  // Verify Results

  const auto are_equal = [](const float a, const float b) {
    return std::abs(a - b) < 0.1f;
  };

  for (int i = 0; i < app_params.m; i++) {
    if (!are_equal(final_results1[i], final_results2[i])) {
      std::cout << "Mismatch found at index " << i << ": " << final_results1[i]
                << " vs. " << final_results2[i] << std::endl;
    }
  }

  rdc::Release();
  return EXIT_SUCCESS;
}
