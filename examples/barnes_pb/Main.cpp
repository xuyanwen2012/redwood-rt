#include <algorithm>
#include <array>
#include <numeric>
#include <queue>
#include <vector>

#include "../LoadFile.hpp"
#include "../Utils.hpp"
#include "Kernel.hpp"
#include "Octree.hpp"
#include "ReducerHandler.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Point.hpp"

std::vector<float> final_results;
constexpr int tid = 0;
int cur_stream = 0;

struct ExecutorStats {
  int leaf_node_reduced = 0;
  int branch_node_reduced = 0;
};

template <typename T>
class Block {
 public:
  Block(int size) { size_ = size; }

  void add(T task) {
    tasks_.push_back(task);
    size_ += 1;
  }

  void recycle() {
    size_ = 0;
    tasks_.clear();
  }

  T get(int index) { return tasks_[index]; }

  int size() { return size_; }

  bool isFull(int block_size) { return size_ >= block_size; }

 private:
  int size_;
  std::vector<T> tasks_;
};

template <typename T>
class BlockSet {
 public:
  BlockSet<T>(int block_size) {
    block_ = nullptr;
    next_block_ = new Block<T>(block_size);
  }
  void setBlock(Block<T>* block) { block_ = block; }

  void setNextBlock(Block<T>* block) { next_block_ = block; }

  Block<T>* getBlock() { return block_; }

  Block<T>* getNextBlock() { return next_block_; }

 private:
  Block<T>* block_;
  Block<T>* next_block_;
};

template <typename T>
class BlockStack {
 public:
  BlockStack(int block_size, int level) {
    stack_ = std::vector<BlockSet<T>>();
    for (int i = 0; i < level; ++i) {
      stack_.push_back(BlockSet<T>(block_size));
    }
  }

  void setBlock(int num, Block<T>* block) { stack_[num].setBlock(block); }

  BlockSet<T> get(int level) { return stack_[level]; }

 private:
  std::vector<BlockSet<T>> stack_;
};

// BH Algorithm
class Executor {
 public:
  Executor(const int tid, const int stream_id)
      : my_tid_(tid), my_stream_id_(stream_id) {}

  static void SetThetaValue(const float theta) { theta_ = theta; }

  void NewTask(const Point4F q) {
    stats_.leaf_node_reduced = 0;
    stats_.branch_node_reduced = 0;
    cached_result_ = 0.0f;
    my_q_ = q;
    rdc::SetQuery(my_tid_, my_stream_id_, q);
  }

  void TraverseRecursiveCpu(const oct::Node<float>* cur) {
    if (cur->IsLeaf()) {
      if (cur->bodies.empty()) return;
      cached_result_ += ReduceLeafsCpu(my_q_, cur->uid);
      ++stats_.leaf_node_reduced;
    } else {
      if (const auto my_theta = ComputeThetaValue(cur, my_q_);
          my_theta < theta_) {
        ++stats_.branch_node_reduced;
        // cached_result_ += KernelFunc(cur->CenterOfMass(), my_q_);
      } else
        for (const auto child : cur->children)
          if (child != nullptr) TraverseRecursiveCpu(child);
    }
  }

  // Main Barnes-Hut Traversal Algorithm, annotated with Redwood APIs
  template <typename T>
  void TraverseRecursive(const oct::Node<float>* cur, BlockStack<T>* stack,
                         int level) {
    BlockSet<T> bset = stack->get(level);
    Block<T>* block = bset.getBlock();
    Block<T>* next_block = bset.getNextBlock();
    next_block->recycle();
    int size = block->size();

    for (int i = 0; i < size; i++) {
      Point4F query = block->get(i);
      NewTask(query);

      if (cur->IsLeaf()) {
        if (cur->bodies.empty()) continue;
        rdc::ReduceLeafNode(my_tid_, my_stream_id_, cur->uid);
        ++stats_.leaf_node_reduced;
        // Read results that were computed and synchronized before

        // Todo: this q_idx is not true, should be the last one
      } else if (const auto my_theta = ComputeThetaValue(cur, query);
                 my_theta < theta_) {
        ++stats_.branch_node_reduced;
        rdc::ReduceBranchNode(my_tid_, my_stream_id_, cur->CenterOfMass());
        // Read results that were computed and synchronized before

        // Todo: this q_idx is not true, should be the last one
      } else {
        next_block->add(query);
      }

      // rdc::LuanchKernelAsync(tid, cur_stream);
    }

    if (next_block->size() > 0) {
      stack->setBlock(level + 1, next_block);
      for (const auto child : cur->children) {
        if (child != nullptr) TraverseRecursive(child, stack, level + 1);
      }
    }
  }

  ExecutorStats GetStats() const { return stats_; }

  float GetResult() const { return cached_result_; }

 protected:
  static float ComputeThetaValue(const oct::Node<float>* node,
                                 const Point4F pos) {
    const auto com = node->CenterOfMass();
    auto norm_sqr = 1e-9f;

    // Use only the first three property (x, y, z) for this theta compuation
    for (int i = 0; i < 3; ++i) {
      const auto diff = com.data[i] - pos.data[i];
      norm_sqr += diff * diff;
    }

    const auto norm = sqrtf(norm_sqr);
    return node->bounding_box.dimension.data[0] / norm;
  }

  static float ReduceLeafsCpu(const Point4F& q, int leaf_idx) {
    constexpr auto leaf_size = 64;
    const auto addr = rdc::LntDataAddr() + leaf_idx * leaf_size;

    float sum{};
    for (int i = 0; i < leaf_size; ++i) {
      sum += KernelFunc(addr[i], q);
    }

    return sum;
  }

 private:
  // Store some reference used
  const int my_tid_;
  const int my_stream_id_;
  static float theta_;  // = 0.2f;
  ExecutorStats stats_;

  Point4F my_q_;

 public:
  // Used on the CPU side
  float cached_result_;
};

float Executor::theta_ = 0.2f;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "requires an input file (\"data/input_bh_2m_4f.dat\")\n";
    return EXIT_FAILURE;
  }

  const char* data_file = argv[1];
  const auto leaf_size = 64;
  const auto theta = 0.2f;

  const auto [in, n] = mmap_file<Point4F>(data_file);

  // Inspect input data is correct
  for (int i = 0; i < 10; ++i) {
    std::cout << in[i] << std::endl;
  }

  std::cout << "Building Tree..." << std::endl;

  const oct::BoundingBox<float> universe{
      Point3F{1000.0f, 1000.0f, 1000.0f},
      Point3F{500.0f, 500.0f, 500.0f},
  };

  const oct::OctreeParams<float> params{theta, leaf_size, universe};
  oct::Octree<float> tree(in, static_cast<int>(n), params);

  tree.BuildTree();

  std::cout << "Loading USM leaf node data..." << std::endl;

  // Initialize Backend, Reducer(double buffer), and Executor(tree algorithm)
  rdc::InitReducers();
  Executor::SetThetaValue(theta);

  // Now octree tree is comstructed, need to move leaf node data into USM
  const auto num_leaf_nodes = tree.GetStats().num_leaf_nodes;

  rdc::AllocateLeafNodeTable(num_leaf_nodes, leaf_size, true);
  tree.LoadPayload(rdc::LntDataAddr(), rdc::LntSizesAddr());

  munmap_file(in, n);

  std::cout << "Making tasks" << std::endl;

  static auto rand_point4f = []() {
    return Point4F{MyRand(0.0f, 1000.0f), MyRand(0.0f, 1000.0f),
                   MyRand(0.0f, 1000.0f), 1.0f};
  };
  const auto m = 32;
  std::queue<Point4F> q_data;
  for (int i = 0; i < m; ++i) q_data.push(rand_point4f());

  // std::vector<float> final_results;
  final_results.reserve(m + 1);  // need to discard the first

  // TimeTask("Cpu Traversal", [&] {
  //   Executor cpu_exe(0, 0);
  //   while (!q_data.empty()) {
  //     const auto q = q_data.front();

  //     cpu_exe.NewTask(q);
  //     cpu_exe.TraverseRecursiveCpu(tree.GetRoot());
  //     final_results.push_back(cpu_exe.cached_result_);

  //     q_data.pop();
  //   }
  // });

  // for (int i = 0; i < m; ++i) {
  //   const auto q = final_results[i];
  //   std::cout << i << ": " << q << std::endl;
  // }

  // Assume in future version this tid will be generated?
  std::cout << "Start Traversal " << std::endl;
  Executor exe[rdc::kNumStreams]{{tid, 0}, {tid, 1}};

  TimeTask("Traversal", [&] {
    Block<Point4F>* blocks[rdc::kNumStreams]{{new Block<Point4F>(2)},
                                             {new Block<Point4F>(2)}};
    // Block<Point4F>* block = new Block<Point4F>(2);
    int level = 0;
    BlockStack<Point4F>* block_stack[rdc::kNumStreams] = {
        new BlockStack<Point4F>(2, tree.GetStats().max_depth + 1),
        new BlockStack<Point4F>(2, tree.GetStats().max_depth + 1)};

    while (!q_data.empty()) {
      const auto q = q_data.front();
      blocks[cur_stream]->add(q);
      // std::cout<<"add"<<std::endl;
      // std::cout<<"size: " <<block->size()<<std::endl;
      // std::cout << "Processing task: " << q << std::endl;
      // Set anchor
      // exe[cur_stream].NewTask(q);

      if (blocks[cur_stream]->isFull(2)) {
        // std::cout<<"full!"<<std::endl;
        block_stack[cur_stream]->setBlock(0, blocks[cur_stream]);
        // Traverse tree and collect data
        exe[cur_stream].TraverseRecursive<Point4F>(
            tree.GetRoot(), block_stack[cur_stream], level);
        std::cout << "\tl: " << exe[cur_stream].GetStats().leaf_node_reduced
                  << "\tb: " << exe[cur_stream].GetStats().branch_node_reduced
                  << std::endl;

        // rdc::LuanchKernelAsync(tid, cur_stream);
        blocks[cur_stream]->recycle();
      }

      // Synchronize the next stream
      const auto next = rdc::NextStream(cur_stream);
      redwood::DeviceStreamSynchronize(next);

      // Read results that were computed and synchronized before
      const auto result = rdc::GetResultValueUnchecked<float>(tid, next);

      // Todo: this q_idx is not true, should be the last one
      final_results.push_back(result);

      // Switch buffer ( A->B, B-A)
      cur_stream = next;
      rdc::ClearBuffer(tid, cur_stream);
      q_data.pop();
    }

    // When q are finished, remember to Synchronize the last bit;
    redwood::DeviceSynchronize();
    const auto next = rdc::NextStream(cur_stream);

    const auto result = rdc::GetResultValueUnchecked<float>(tid, next);
    final_results.push_back(result);
  });

  for (int i = 0; i < m; ++i) {
    const auto q = final_results[i + 1];
    std::cout << i << ": " << q << std::endl;
  }

  rdc::ReleaseReducers();
  return EXIT_SUCCESS;
}