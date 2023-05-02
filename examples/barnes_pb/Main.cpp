#include <omp.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <queue>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "../barnes/AppParams.hpp"
#include "../cxxopts.hpp"
#include "Functors/DistanceMetrics.hpp"
#include "Octree.hpp"
#include "ReducerHandler.hpp"
#include "Redwood.hpp"

using Task = std::pair<int, Point4F>;

std::vector<float> final_results;
/*
template <typename T>
struct QueryNode{
  T query;
  int q_idx;
};
*/

class Block {
 public:
  Block(int size) { size_ = size; }

  void add(Task task) {
    tasks_.push_back(task);
    size_ += 1;
  }

  void recycle() {
    size_ = 0;
    tasks_.clear();
  }

  Task get(int index) { return tasks_[index]; }

  int size() { return size_; }

  bool isFull(int block_size) { return size_ >= block_size; }

 private:
  int size_;
  std::vector<Task> tasks_;
};

class BlockSet {
 public:
  BlockSet(int block_size) {
    block_ = nullptr;
    next_block_ = new Block(block_size);
  }
  void setBlock(Block* block) { block_ = block; }

  void setNextBlock(Block* block) { next_block_ = block; }

  Block* getBlock() { return block_; }

  Block* getNextBlock() { return next_block_; }

 private:
  Block* block_;
  Block* next_block_;
};

class BlockStack {
 public:
  BlockStack(int block_size, int level) {
    stack_ = std::vector<BlockSet>();
    for (int i = 0; i < level; ++i) {
      stack_.push_back(BlockSet(block_size));
    }
  }

  void setBlock(int num, Block* block) { stack_[num].setBlock(block); }

  BlockSet get(int level) { return stack_[level]; }

 private:
  std::vector<BlockSet> stack_;
};

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

// Traverser class for BH Algorithm
class Executor {
  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;
  ExecutorStats stats_;

  Point4F my_q_;

  // Used on the CPU side
  float host_result_;
  // newly added
  const int my_block_size_;

 public:
  Executor(const int tid, const int stream_id, const int block_size)
      : my_tid_(tid), my_stream_id_(stream_id), my_block_size_(block_size) {}

  /*
  void StartQuery(const Point4F q, const oct::Node<float>* root) {
    // Clear executor's data
    stats_.leaf_node_reduced = 0;
    stats_.branch_node_reduced = 0;

    // This is a host copy of q point, so some times the traversal doesn't have
    // to bother with accessing USM
    my_q_ = q;

    // Notify Reducer to
    // In case of FPGA, it will register the anchor point (q) into the registers
    rdc::SetQuery(my_tid_, my_stream_id_, my_q_);

    TraversePointBlocking(root);
  }
  */

  void StartQueryCpu(const Point4F q, const oct::Node<float>* root) {
    stats_.leaf_node_reduced = 0;
    stats_.branch_node_reduced = 0;
    my_q_ = q;
    host_result_ = 0.0f;
    // host_results_ = std::unordered_map<int, float>{};
    TraverseRecursiveCpu(root);
  }

  void StartQueryPb(const Point4F q, const oct::Node<float>* root,
                    BlockStack* stack) {
    stats_.leaf_node_reduced = 0;
    stats_.branch_node_reduced = 0;
    my_q_ = q;
    host_result_ = 0.0f;
    // host_results_ = std::unordered_map<int, float>{};
    // std::cout <<"traverse point blocking"<<std::endl;
    TraversePointBlocking(root, stack, 0);
  }

  _NODISCARD ExecutorStats GetStats() const { return stats_; }

  _NODISCARD float GetCpuResult() const { return host_result_; }
  // _NODISCARD std::unordered_map<int, float> GetCpuPbResult() const {return
  // host_results_;}
 private:
  _NODISCARD static float ComputeThetaValue(const oct::Node<float>* node,
                                            const Point4F& pos) {
    //   std::cout << "compute thetavalue" <<std::endl;
    const auto com = node->CenterOfMass();
    //  std::cout <<"done getting center of mass"<<std::endl;
    auto norm_sqr = 1e-9f;

    // Use only the first three property (x, y, z) for this theta compuation
    for (int i = 0; i < 3; ++i) {
      const auto diff = com.data[i] - pos.data[i];
      norm_sqr += diff * diff;
    }

    const auto norm = sqrtf(norm_sqr);
    return node->bounding_box.dimension.data[0] / norm;
  }

  void TraversePointBlocking(const oct::Node<float>* cur, BlockStack* stack,
                             int level) {
    constexpr dist::Gravity functor;
    BlockSet bset = stack->get(level);
    Block* block = bset.getBlock();
    Block* next_block = bset.getNextBlock();
    next_block->recycle();
    int size = block->size();
    for (int i = 0; i < size; i++) {
      Task query_node = block->get(i);
      if (cur->IsLeaf()) {
        if (cur->bodies.empty()) {
          return;
        }

        // ------------------------------------------------------------
        const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
        for (int j = 0; j < app_params.max_leaf_size; ++j) {
          host_result_ += functor(query_node.second, leaf_addr[j]);
          // final_results[query_node.q_idx] += functor(query_node.query,
          // leaf_addr[j]);
        }
        // ------------------------------------------------------------

        ++stats_.leaf_node_reduced;
      } else if (const auto my_theta =
                     ComputeThetaValue(cur, query_node.second);
                 my_theta < app_params.theta) {
        ++stats_.branch_node_reduced;
        // ------------------------------------------------------------
        host_result_ += functor(query_node.second, cur->CenterOfMass());
        // final_results[query_node.q_idx] += functor(query_node.query,
        // cur->CenterOfMass()); [query_node.q_idx] += functor(query_node.query,
        // cur->CenterOfMass());
        // ---------------------------------------------------------------

      } else {
        next_block->add(query_node);
      }
    }
    if (next_block->size() > 0) {
      stack->setBlock(level + 1, next_block);
      for (const auto child : cur->children) {
        if (child != nullptr) TraversePointBlocking(child, stack, level + 1);
      }
    }
  }
  // CPU version
  void TraverseRecursiveCpu(const oct::Node<float>* cur) {
    constexpr dist::Gravity functor;

    if (cur->IsLeaf()) {
      if (cur->bodies.empty()) return;

      // ------------------------------------------------------------
      const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
      for (int i = 0; i < app_params.max_leaf_size; ++i) {
        host_result_ += functor(my_q_, leaf_addr[i]);
      }
      // ------------------------------------------------------------

      ++stats_.leaf_node_reduced;
    } else if (const auto my_theta = ComputeThetaValue(cur, my_q_);
               my_theta < app_params.theta) {
      ++stats_.branch_node_reduced;

      // ------------------------------------------------------------
      host_result_ += functor(my_q_, cur->CenterOfMass());
      // ---------------------------------------------------------------

    } else
      for (const auto child : cur->children)
        if (child != nullptr) TraverseRecursiveCpu(child);
  }
};

_NODISCARD inline Point4F RandPoint() {
  Point4F p;
  p.data[0] = MyRand(0, 1000);
  p.data[1] = MyRand(0, 1000);
  p.data[2] = MyRand(0, 1000);
  p.data[3] = MyRand(0, 1000);
  return p;
}

int main(int argc, char** argv) {
  cxxopts::Options options("Barnes-Hut (BH)", "Redwood BH demo implementation");

  // clang-format off
  options.add_options()
    ("f,file", "Input file name", cxxopts::value<std::string>())
    ("m,query", "Number of particles to query", cxxopts::value<int>()->default_value("1048576"))
    ("t,thread", "Number of threads", cxxopts::value<int>()->default_value("1"))
    ("theta", "Theta Value", cxxopts::value<float>()->default_value("0.2"))
    ("l,leaf", "Maximum leaf node size", cxxopts::value<int>()->default_value("32"))
    ("b,batch_size", "Batch size (GPU)", cxxopts::value<int>()->default_value("2048"))
    ("c,cpu", "Enable CPU baseline", cxxopts::value<bool>()->default_value("false"))
    ("h,help", "Print usage");
  // clang-format on

  options.parse_positional({"file", "query"});

  const auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help() << std::endl;
    exit(EXIT_SUCCESS);
  }

  if (!result.count("file")) {
    std::cerr << "requires an input file (\"../../data/2m_bh_uniform.dat\")\n";
    std::cout << options.help() << std::endl;
    exit(EXIT_FAILURE);
  }

  const auto data_file = result["file"].as<std::string>();
  app_params.m = result["query"].as<int>();
  app_params.num_threads = result["thread"].as<int>();
  app_params.theta = result["theta"].as<float>();
  app_params.max_leaf_size = result["leaf"].as<int>();
  app_params.batch_size = result["batch_size"].as<int>();
  app_params.cpu = result["cpu"].as<bool>();
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

  std::vector<Task> temp_vec(q_data[0].size());
  for (std::size_t i = 0; i < temp_vec.size(); i++) {
    temp_vec[i] = q_data[0].front();
    q_data[0].pop();
  }

  std::sort(temp_vec.begin(), temp_vec.end(),
            [](const auto& lhs, const auto& rhs) {
              Point4F lhp = lhs.second;
              Point4F rhp = rhs.second;
              if (lhp.data[0] != rhp.data[0]) {
                return lhp.data[0] < rhp.data[0];
              }
              if (lhp.data[1] != rhp.data[1]) {
                return lhp.data[1] < rhp.data[1];
              }
              if (lhp.data[2] != rhp.data[2]) {
                return lhp.data[2] < rhp.data[2];
              }
              return lhp.data[3] < rhp.data[3];
            });

  for (const auto& elem : temp_vec) {
    q_data[0].push(elem);
  }
  std::cout << "Building Tree..." << std::endl;

  const oct::BoundingBox<float> universe{
      Point3F{1000.0f, 1000.0f, 1000.0f},
      Point3F{500.0f, 500.0f, 500.0f},
  };
  const oct::OctreeParams<float> params{
      app_params.theta, static_cast<size_t>(app_params.max_leaf_size),
      universe};
  oct::Octree<float> tree(in_data.data(), static_cast<int>(n), params);
  tree.BuildTree();

  // Init
  rdc::Init(app_params.num_threads, app_params.batch_size);
  omp_set_num_threads(app_params.num_threads);

  const auto num_leaf_nodes = tree.GetStats().num_leaf_nodes;
  auto [lnt_addr, lnt_size_addr] =
      rdc::AllocateLnt(num_leaf_nodes, app_params.max_leaf_size);

  tree.LoadPayload(lnt_addr, lnt_size_addr);

  final_results.resize(app_params.m);  // need to discard the first

  std::cout << "Starting Traversal... " << std::endl;

  TimeTask("Traversal", [&] {
    if (app_params.cpu) {
      // ------------------- CPU ------------------------------------
      std::vector<Executor> cpu_exe;
      std::vector<Block*> blocks;
      std::vector<BlockStack*> block_stack;
      int level = 0;
      const int block_size = 512;

      for (int tid = 0; tid < app_params.num_threads; ++tid) {
        cpu_exe.emplace_back(tid, 0, block_size);
        blocks.push_back(new Block(block_size));
        block_stack.push_back(
            new BlockStack(block_size, tree.GetStats().max_depth + 1));
      }

      // Run
#pragma omp parallel for
      for (int tid = 0; tid < app_params.num_threads; ++tid) {
        // std::cout << "tid: "<<tid<<std::endl;
        while (!q_data[tid].empty()) {
          const auto [q_idx, q] = q_data[tid].front();
          blocks[tid]->add(Task{q_idx, q});
          q_data[tid].pop();
          if (blocks[tid]->isFull(block_size)) {
            block_stack[tid]->setBlock(0, blocks[tid]);
            cpu_exe[tid].StartQueryPb(q, tree.GetRoot(), block_stack[tid]);
            blocks[tid]->recycle();
          }
        }
      }

      // -------------------------------------------------------------
    } else {
      // ------------------- CUDA ------------------------------------

      // -------------------------------------------------------------
    }
  });

  // -------------------------------------------------------------
  for (int i = 0; i < 5; ++i) {
    const auto q = final_results[i];
    std::cout << i << ": " << q << std::endl;
  }

  rdc::Release();
  return EXIT_SUCCESS;
}