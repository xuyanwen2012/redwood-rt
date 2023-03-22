#include <algorithm>
#include <array>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <numeric>
#include <queue>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "../cxxopts.hpp"
#include "../knn/KDTree.hpp"
#include "../knn/Kernel.hpp"
#include "../knn/KnnSet.hpp"
#include "AppParams.hpp"
#include "ReducerHandler.hpp"
#include "Redwood/Core.hpp"

inline std::shared_ptr<kdt::KdTree> tree_ref;

enum class ExecutionState { kWorking, kFinished };

struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

// Knn Algorithm
class Executor {
 public:
  // Executor() = delete;

  // Thread id, i.e., [0, .., n_threads]
  // Stream id in the thread, i.e., [0, 1]
  // My id in the group executor, i.e., [0,...,1023]
  Executor(const int tid, const int stream_id, const int my_id)
      : state_(ExecutionState::kFinished),
        my_tid_(tid),
        my_stream_id_(stream_id),
        debug_uid_(my_id) {
    // values need experiment
    stack_.reserve(16);

    // When created,
    float* base_addr = rdc::GetResultAddr<float>(my_tid_, my_stream_id_);
    u_my_result_addr_ = base_addr + my_id;
  }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

  void SetQuery(const Point4F& q) { my_query_point_ = q; }

  void StartQuery() {
    stack_.clear();
    k_set_->Clear();
    cur_ = nullptr;
    Execute();
  }

  void Resume() { Execute(); }

  void CPUTraverse() {
    k_set_->Clear();
    TraversalRecursive(tree_ref->root_);
  }

 protected:
  void Execute() {
    if (state_ == ExecutionState::kWorking) goto my_resume_point;
    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          //
          // **** Reduction at Leaf Node (replaced with Redwood API) ****
          rdc::ReduceLeafNode(my_tid_, my_stream_id_, cur_->uid,
                              my_query_point_);
          // ****************************

          // **** Coroutine Reuturn (API) ****
          return;
        my_resume_point:
          // ****************************

          cur_ = nullptr;
          continue;
        }

        // **** Reduction at tree node ****
        const unsigned accessor_idx =
            tree_ref->v_acc_[cur_->node_type.tree.idx_mid];

        const float dist =
            KernelFunc(tree_ref->in_data_ref_[accessor_idx], my_query_point_);

        k_set_->Insert(dist);

        // **********************************

        const int axis = cur_->node_type.tree.axis;
        const float train = tree_ref->in_data_ref_[accessor_idx].data[axis];
        const kdt::Dir dir = my_query_point_.data[axis] < train
                                 ? kdt::Dir::kLeft
                                 : kdt::Dir::kRight;

        stack_.push_back({cur_, axis, train, dir});
        cur_ = cur_->GetChild(dir);
      }

      // We resume back from Break point, and now we are still in the branch
      // node, we can check if there's any thing left on the stack.
      if (!stack_.empty()) {
        const auto [last_cur, axis, train, dir] = stack_.back();
        stack_.pop_back();

        // Check if there is a possibility of the NN lies on the other half
        // If the difference between the query point and the other splitting
        // plane is greater than the current found minimum distance, then it
        // is impossible to have a NN there.

        Point4F a{};
        Point4F b{};
        a.data[axis] = my_query_point_.data[axis];
        b.data[axis] = train;
        const auto diff = KernelFunc(a, b);
        if (diff < k_set_->WorstDist()) {
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals
    state_ = ExecutionState::kFinished;
  }

  void TraversalRecursive(const kdt::Node* cur) {
    if (cur->IsLeaf()) {
      const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
      for (int i = 0; i < tree_ref->GetParams().leaf_max_size; ++i) {
        const float dist = KernelFunc(leaf_addr[i], my_query_point_);
        k_set_->Insert(dist);
      }

    } else {
      // **** Reduction at tree node ****
      const unsigned accessor_idx =
          tree_ref->v_acc_[cur->node_type.tree.idx_mid];
      const float dist =
          KernelFunc(tree_ref->in_data_ref_[accessor_idx], my_query_point_);
      k_set_->Insert(dist);

      // **********************************

      const auto axis = cur->node_type.tree.axis;
      const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
      const auto dir = my_query_point_.data[axis] < train ? kdt::Dir::kLeft
                                                          : kdt::Dir::kRight;

      TraversalRecursive(cur->GetChild(dir));

      Point4F a{};
      Point4F b{};
      a.data[axis] = my_query_point_.data[axis];
      b.data[axis] = train;
      const auto diff = KernelFunc(a, b);
      if (diff < k_set_->WorstDist()) {
        TraversalRecursive(cur->GetChild(FlipDir(dir)));
      }
    }
  }

 private:
  Point4F my_query_point_;

  // this should point to USM region, unique to each executor.
  // If k = 32,
  // Each address is 128 bytes (or 16 eight-byte words) apart.
 public:
  // Pointer to the USM address requested from the Backend.
  union {
    KnnSet<float, 1>* k_set_;
    float* u_my_result_addr_;
  };

  // Couroutine related
  std::vector<CallStackField> stack_;
  kdt::Node* cur_;
  ExecutionState state_;

  // Store some reference used
  int my_tid_;
  int my_stream_id_;

 public:
  int debug_uid_;
};

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
  app_params.batch_size = result["batch_size"].as<int>();
  app_params.num_threads = result["thread"].as<int>();
  app_params.m = result["query"].as<int>();
  app_params.cpu = result["cpu"].as<bool>();

  // Load file
  const auto [in, n] = mmap_file<Point4F>(data_file);

  std::cout << app_params << std::endl;

  // Inspect input data is correct
  for (int i = 0; i < 10; ++i) {
    std::cout << in[i] << std::endl;
  }

  std::cout << "Building Tree..." << std::endl;
  const kdt::KdtParams params{app_params.max_leaf_size};
  auto tree = std::make_shared<kdt::KdTree>(params, in, n);
  tree_ref = tree;

  std::cout << "Loading USM leaf node data..." << std::endl;
  rdc::InitReducers();

  // Now octree tree is comstructed, need to move leaf node data into USM
  const auto num_leaf_nodes = tree->GetStats().num_leaf_nodes;
  rdc::AllocateLeafNodeTable(num_leaf_nodes, app_params.max_leaf_size, false);
  tree->LoadPayload(rdc::LntDataAddr());

  // TODO: figure out a better way, not important right now
  // munmap_file(in, n);

  std::cout << "Making tasks..." << std::endl;

  static auto rand_point4f = []() {
    // return Point4F{MyRand(0.0f, 100.0f), MyRand(0.0f, 100.0f),
    //  MyRand(0.0f, 100.0f), MyRand(0.0f, 100.0f)};
    return Point4F{MyRand(0.0f, 1000.0f), MyRand(0.0f, 1000.0f),
                   MyRand(0.0f, 1000.0f), MyRand(0.0f, 1000.0f)};
  };

  std::queue<Point4F> q_data;
  for (int i = 0; i < app_params.m; ++i) q_data.push(rand_point4f());

  std::cout << "Start Traversal " << std::endl;

  constexpr int tid = 0;

  std::vector<float> final_results;
  final_results.reserve(app_params.m);

  if (app_params.cpu) {
    TimeTask("Traversal", [&] {
      Executor exe{tid, 0, 0};
      while (!q_data.empty()) {
        const auto q = q_data.front();
        q_data.pop();
        exe.SetQuery(q);
        exe.CPUTraverse();
        final_results.push_back(exe.k_set_->WorstDist());
      }
    });

  } else {
    TimeTask("Traversal", [&] {
      // Use redwood runtime
      std::vector<Executor> exes;
      exes.reserve(app_params.batch_size);
      for (int i = 0; i < app_params.batch_size; ++i)
        exes.emplace_back(tid, 0, i);

      for (auto& exe : exes) {
        if (q_data.empty()) break;
        const auto q = q_data.front();
        q_data.pop();
        exe.SetQuery(q);
        exe.StartQuery();
      }

      rdc::LuanchKernelAsync(tid, 0);
      redwood::DeviceStreamSynchronize(0);
      rdc::ClearBuffer(tid, 0);

      while (!q_data.empty()) {
        for (auto& exe : exes) {
          if (exe.Finished()) {
            final_results.push_back(exe.k_set_->WorstDist());

            // Make there is task in the queue
            if (q_data.empty()) {
              break;
            }
            const auto q = q_data.front();
            q_data.pop();
            exe.SetQuery(q);
            exe.StartQuery();

          } else {
            exe.Resume();
          }
        }

        rdc::LuanchKernelAsync(tid, 0);
        redwood::DeviceStreamSynchronize(0);
        rdc::ClearBuffer(tid, 0);
      }

      // Now there's still remaining
      for (auto& exe : exes) {
        exe.CPUTraverse();
        final_results.push_back(exe.k_set_->WorstDist());
      }
    });

    // TimeTask("Traversal", [&] {
    //   // Use redwood runtime
    //   std::vector<Executor> exes[rdc::kNumStreams];
    //   for (int stream_id = 0; stream_id < rdc::kNumStreams; ++stream_id) {
    //     exes[stream_id].reserve(app_params.batch_size);
    //     for (int i = 0; i < app_params.batch_size; ++i) {
    //       exes[stream_id].emplace_back(tid, stream_id, i);
    //     }
    //   }

    //   if constexpr (false) {
    //     for (int stream_id = 0; stream_id < rdc::kNumStreams; ++stream_id)
    //       for (auto& exe : exes[stream_id])
    //         std::cout << exe.my_tid_ << "-" << exe.my_stream_id_ << "-"
    //                   << exe.debug_uid_ << "\t" << exe.u_my_result_addr_
    //                   << std::endl;
    //   }

    //   // During the initial iteration, we can assume they are all finished
    //   // Batch A, first pass
    //   for (auto& exe : exes[0]) {
    //     if (q_data.empty()) break;
    //     const auto q = q_data.front();
    //     q_data.pop();
    //     exe.SetQuery(q);
    //     exe.StartQuery();
    //   }
    //   rdc::LuanchKernelAsync(tid, 0);

    //   // Batch B, first pass
    //   for (auto& exe : exes[1]) {
    //     if (q_data.empty()) break;
    //     const auto q = q_data.front();
    //     q_data.pop();
    //     exe.SetQuery(q);
    //     exe.StartQuery();
    //   }
    //   rdc::LuanchKernelAsync(tid, 1);

    //   redwood::DeviceStreamSynchronize(0);
    //   rdc::ClearBuffer(tid, 0);

    //   // Batch A or B, alternating
    //   auto cur_stream = 0;
    //   while (!q_data.empty()) {
    //     for (auto& exe : exes[cur_stream]) {
    //       if (exe.Finished()) {
    //         final_results.push_back(exe.k_set_->WorstDist());

    //         // Make there is task in the queue
    //         if (q_data.empty()) {
    //           break;
    //         }

    //         const auto q = q_data.front();
    //         q_data.pop();
    //         exe.SetQuery(q);
    //         exe.StartQuery();
    //       } else {
    //         exe.Resume();
    //       }
    //     }

    //     rdc::LuanchKernelAsync(tid, cur_stream);

    //     // Switch buffer ( A->B, B-A)
    //     const auto next = rdc::NextStream(cur_stream);
    //     redwood::DeviceStreamSynchronize(next);
    //     cur_stream = next;
    //     rdc::ClearBuffer(tid, cur_stream);
    //   }

    //   // Tasks are empty, but both Batch still not empty yet.

    //   // while (!exes[0].empty() || !exes[1].empty()) {
    //   //   auto it = exes[cur_stream].begin();
    //   //   std::cout << "[stream " << cur_stream
    //   //             << "] Now processing the remaining: "
    //   //             << std::distance(it, std::end(exes[cur_stream])) <<
    //   //             std::endl;

    //   //   constexpr auto threshold = 64;
    //   //   if (std::distance(it, std::end(exes[cur_stream])) < threshold) {
    //   //     // Too small to offload to GPU, just do them on CPU

    //   //     // redwood::DeviceStreamSynchronize(next);
    //   //     for (auto& exe : exes[0]) {
    //   //       exe.CPUTraverse();
    //   //       final_results.push_back(exe.k_set_->WorstDist());
    //   //     }
    //   //     for (auto& exe : exes[1]) {
    //   //       exe.CPUTraverse();
    //   //       final_results.push_back(exe.k_set_->WorstDist());
    //   //     }

    //   //     break;

    //   //   } else {
    //   //     while (it != exes[cur_stream].end()) {
    //   //       if (it->Finished()) {
    //   //         final_results.push_back(it->k_set_->WorstDist());
    //   //         it = exes[cur_stream].erase(it);
    //   //       } else {
    //   //         it->Resume();
    //   //         ++it;
    //   //       }
    //   //     }

    //   //     rdc::LuanchKernelAsync(tid, cur_stream);
    //   //     const auto next = rdc::NextStream(cur_stream);
    //   //     redwood::DeviceStreamSynchronize(next);
    //   //     cur_stream = next;
    //   //     rdc::ClearBuffer(tid, cur_stream);
    //   //   }
    //   // }

    //   // exit(0);
    // });
  }

  std::sort(final_results.begin(), final_results.end());
  for (int i = 0; i < 32; ++i) {
    const auto q = final_results[i];
    std::cout << "final " << i << ": " << q << std::endl;
  }

  std::cout << "final_results.size() " << final_results.size() << std::endl;

  rdc::ReleaseReducers();
  return EXIT_SUCCESS;
}