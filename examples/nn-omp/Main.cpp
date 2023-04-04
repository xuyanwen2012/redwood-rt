#include <omp.h>

#include <algorithm>
#include <cassert>
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
#include "../nn/GlobalVars.hpp"
#include "../nn/KDTree.hpp"
#include "Executor.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "ReducerHandler.hpp"
#include "Redwood.hpp"

_NODISCARD inline Point4F RandPoint() {
  Point4F p;
  p.data[0] = MyRand(0, 1024);
  p.data[1] = MyRand(0, 1024);
  p.data[2] = MyRand(0, 1024);
  p.data[3] = MyRand(0, 1024);
  return p;
}

int main(int argc, char** argv) {
  cxxopts::Options options("Nearest Neighbor (NN)",
                           "Redwood NN demo implementation");

  // clang-format off
  options.add_options()
    ("f,file", "Input file name", cxxopts::value<std::string>())
    ("m,query", "Number of particles to query", cxxopts::value<int>()->default_value("1048576"))
    ("t,thread", "Number of threads", cxxopts::value<int>()->default_value("1"))
    ("l,leaf", "Maximum leaf node size", cxxopts::value<int>()->default_value("32"))
    ("b,batch_size", "Batch size (GPU)", cxxopts::value<int>()->default_value("1024"))
    ("c,cpu", "Enable CPU baseline", cxxopts::value<bool>()->default_value("false"))
    ("d,dump", "Dump result to a tem file", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage");
  // clang-format on

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
  app_params.m = result["query"].as<int>();
  app_params.num_threads = result["thread"].as<int>();
  app_params.max_leaf_size = result["leaf"].as<int>();
  app_params.batch_size = result["batch_size"].as<int>();
  app_params.cpu = result["cpu"].as<bool>();
  app_params.dump = result["dump"].as<bool>();
  std::cout << app_params << std::endl;

  std::cout << "Loading Data..." << std::endl;

  const auto in_data = load_data_from_file<Point4F>(data_file);
  const auto n = in_data.size();

  // For each thread
  std::vector<std::queue<Task>> q_data(app_params.num_threads);
  const auto tasks_per_thread = app_params.m / app_params.num_threads;
  for (int tid = 0; tid < app_params.num_threads; ++tid) {
    for (int i = 0; i < tasks_per_thread; ++i) {
      q_data[tid].emplace(i, RandPoint());
    }
  }

  std::cout << "Building kd Tree..." << std::endl;

  const kdt::KdtParams params{app_params.max_leaf_size};
  tree_ref = std::make_shared<kdt::KdTree>(params, in_data.data(), n);

  const auto num_leaf_nodes = tree_ref->GetStats().num_leaf_nodes;
  auto lnt_addr = rdc::AllocateLnt(num_leaf_nodes, app_params.max_leaf_size);
  tree_ref->LoadPayload(lnt_addr);

  // Init
  rdc::Init(app_params.num_threads, app_params.batch_size);
  omp_set_num_threads(app_params.num_threads);
  final_results1.resize(app_params.m);

  std::cout << "Starting Traversal..." << std::endl;
  if (app_params.cpu) {
    TimeTask("CPU Traversal", [&] {
      std::vector<Executor<dist::Euclidean>> cpu_exe;
      for (int tid = 0; tid < app_params.num_threads; ++tid) {
        cpu_exe.emplace_back(tid, 0, 0);
      }

      // Run
#pragma omp parallel for
      for (int tid = 0; tid < app_params.num_threads; ++tid) {
        while (!q_data[tid].empty()) {
          const auto task = q_data[tid].front();
          cpu_exe[tid].SetQuery(task);
          final_results1[task.first] = cpu_exe[tid].CpuTraverse();
          q_data[tid].pop();
        }
      }
    });

  } else {
    // Use Redwood
    constexpr auto num_streams = 2;
    final_results1.resize(app_params.m);

    // Setup traversers
    std::vector<Executor<dist::Euclidean>> exes;
    exes.reserve(app_params.num_threads * num_streams * app_params.batch_size);
    for (int tid = 0; tid < app_params.num_threads; ++tid) {
      for (int stream_id = 0; stream_id < num_streams; ++stream_id) {
        for (int i = 0; i < app_params.batch_size; ++i) {
          exes.emplace_back(tid, stream_id, i);
        }
      }
    }

    // exe[4][2][1024]
    const auto tid_offset = num_streams * app_params.batch_size;
    const auto stream_offset = app_params.batch_size;

    TimeTask("GPU Traversal", [&] {
#pragma omp parallel for
      for (int tid = 0; tid < app_params.num_threads; ++tid) {
        auto cur_stream = 0;
        while (!q_data[tid].empty()) {
          auto it = tid * tid_offset + cur_stream * stream_offset;
          const auto it_end = it + app_params.batch_size;
          for (; it != it_end;) {
            if (exes[it].Finished()) {
              if (!q_data[tid].empty()) {
                const auto q = q_data[tid].front();
                q_data[tid].pop();

                exes[it].SetQuery(q);
                exes[it].StartQuery();
              }

              ++it;
            } else {
              exes[it].Resume();
              if (exes[it].Finished()) {
                // Do not increment 'it'
              } else {
                ++it;
              }
            }
          }

          rdc::LaunchAsyncWorkQueue(tid, cur_stream);

          // switch to next
          cur_stream = (cur_stream + 1) % num_streams;

          redwood::DeviceStreamSynchronize(tid, cur_stream);
          rdc::ResetBuffer(tid, cur_stream);
        }
      }

      redwood::DeviceSynchronize();

      for (int tid = 0; tid < app_params.num_threads; ++tid) {
        for (int cur_stream = 0; cur_stream < num_streams; ++cur_stream) {
          auto it = tid * tid_offset + cur_stream * stream_offset;
          const auto it_end = it + app_params.batch_size;
          for (; it != it_end; ++it) {
            const auto q_idx = exes[it].my_task_.first;
            final_results1[q_idx] = exes[it].CpuTraverse();
          }
        }
      }
    });
  }
  std::cout << "Program Execution Completed. " << std::endl;

  // Peek results
  for (int i = 0; i < 5; ++i) {
    std::cout << final_results1[i] << '\n';
  }
  std::cout << "..." << std::endl;

  if (app_params.dump) {
    DumpFile<float>(final_results1, app_params.cpu);
  }

  rdc::Release();
  return EXIT_SUCCESS;
}
