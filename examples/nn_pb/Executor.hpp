#pragma once

#include "GlobalVars.hpp"
#include "KDTree.hpp"
#include "ReducerHandler.hpp"

using Task = std::pair<int, Point4F>;

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
    next_block_2_ = new Block(block_size);
  }
  void setBlock(Block* block) { block_ = block; }

  void setNextBlock(Block* block) { next_block_ = block; }

  void setNextBlock2(Block* block) { next_block_2_ = block; }

  Block* getBlock() { return block_; }

  Block* getNextBlock() { return next_block_; }

  Block* getNextBlock2() { return next_block_2_; }

 private:
  Block* block_;
  Block* next_block_;
  Block* next_block_2_;
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

enum class ExecutionState { kWorking, kFinished };

struct CallStackField {
  kdt::Node* current;
  int axis;
  float train;
  kdt::Dir dir;
};

// Nn/Knn Algorithm
template <typename Functor>
class Executor {
 public:
  // Thread id, i.e., [0, .., n_threads]
  // Stream id in the thread, i.e., [0, 1]
  // My id in the group executor, i.e., [0,...,1023]
  Executor(const int tid, const int stream_id, const int uid)
      : cur_(),
        state_(ExecutionState::kFinished),
        my_tid_(tid),
        my_stream_id_(stream_id),
        my_uid_(uid) {
    stack_.reserve(16);
    my_assigned_result_addr = rdc::RequestResultAddr(tid, stream_id, uid);
  }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

  void SetQuery(const Task& task) { my_task_ = task; }

  void StartQuery() {
    stack_.clear();
    result_set->Reset();
    Execute();
  }

  void Resume() { Execute(); }

  _NODISCARD float CpuTraverse() {
    result_set->Reset();
    TraversalRecursive(tree_ref->root_);
    return result_set->WorstDist();
  }
    _NODISCARD float CpuTraversePb(BlockStack *block_stack, int level) {
    result_set->Reset();
    TraversalRecursivePB(tree_ref->root_, block_stack, 0);
    return result_set->WorstDist();
  }

 protected:
  void Execute() {
    constexpr Functor functor;

    if (state_ == ExecutionState::kWorking) goto my_resume_point;
    state_ = ExecutionState::kWorking;
    cur_ = tree_ref->root_;

    // Begin Iteration
    while (cur_ != nullptr || !stack_.empty()) {
      // Traverse all the way to left most leaf node
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          // **** Reduction at Leaf Node (replaced with Redwood API) ****

          rdc::ReduceLeafNode(my_tid_, my_stream_id_, my_task_, cur_->uid);

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
            functor(tree_ref->in_data_ref_[accessor_idx], my_task_.second);

        result_set->Insert(dist);
        // **********************************

        // Determine which child node to traverse next
        const auto axis = cur_->node_type.tree.axis;
        const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
        const auto dir = my_task_.second.data[axis] < train ? kdt::Dir::kLeft
                                                            : kdt::Dir::kRight;

        // Recursion 1
        stack_.push_back({cur_, axis, train, dir});
        cur_ = cur_->GetChild(dir);
      }

      if (!stack_.empty()) {
        const auto [last_cur, axis, train, dir] = stack_.back();
        stack_.pop_back();

        if (const auto diff = functor(my_task_.second.data[axis], train);
            diff < result_set->WorstDist()) {
          // Recursion 2
          cur_ = last_cur->GetChild(FlipDir(dir));
        }
      }
    }

    // Done traversals
    state_ = ExecutionState::kFinished;

    final_results1[my_task_.first] = result_set->WorstDist();
  }

  void TraversalRecursivePB(const kdt::Node* cur, BlockStack* block_stack,
                            int level) {
    constexpr Functor functor;
    BlockSet bset = block_stack->get(level);
    Block* block = bset.getBlock();
    Block* next_block = bset.getNextBlock();
    Block* next_block2 = bset.getNextBlock2();
    next_block->recycle();
    next_block2->recycle();
    int size = block->size();
    for (int i = 0; i < size; i++) {
      //std::cout <<"i: " << i <<std::endl;
      Task query_node = block->get(i);
      if (cur->IsLeaf()) {
        // **** Reduction at leaf node ****
        const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
        for (int j = 0; j < rdc::stored_max_leaf_size; ++j) {
          const float dist = functor(leaf_addr[i], query_node.second);
          result_set->Insert(dist);
        }
        // **********************************
      } else {
        // **** Reduction at tree node ****
        const unsigned accessor_idx =
            tree_ref->v_acc_[cur->node_type.tree.idx_mid];
        const float dist =
            functor(tree_ref->in_data_ref_[accessor_idx], query_node.second);
        result_set->Insert(dist);
        // **********************************

        // Determine which child node to traverse next
        const auto axis = cur->node_type.tree.axis;
        const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
        const auto dir = query_node.second.data[axis] < train
                             ? kdt::Dir::kLeft
                             : kdt::Dir::kRight;
        if (dir == kdt::Dir::kLeft) {
          next_block->add(query_node);
        } else {
          next_block2->add(query_node);
        }
        if (const auto diff = functor(query_node.second.data[axis], train);
            diff < result_set->WorstDist()) {
          if (dir == kdt::Dir::kRight) {
            next_block->add(query_node);
          } else {
            next_block2->add(query_node);
          }
        }
      }
    }
    if (next_block->size() > 0) {
      block_stack->setBlock(level + 1, next_block);
      TraversalRecursivePB(cur->GetChild(kdt::Dir::kLeft), block_stack,
                           level + 1);
    }
    if (next_block2->size() > 0) {
      block_stack->setBlock(level + 1, next_block2);
      TraversalRecursivePB(cur->GetChild(kdt::Dir::kRight), block_stack,
                           level + 1);
    }
  }

  void TraversalRecursive(const kdt::Node* cur) {
    constexpr Functor functor;
    if (cur->IsLeaf()) {
      // **** Reduction at leaf node ****
      const auto leaf_addr = rdc::LntDataAddrAt(cur->uid);
      for (int i = 0; i < rdc::stored_max_leaf_size; ++i) {
        const float dist = functor(leaf_addr[i], my_task_.second);
        result_set->Insert(dist);
      }
      // **********************************
    } else {
      auto range = kdt::GetSubRange(cur, my_task_);
      int min = std::get<0>(range);
      int max = std::get<1>(range);
      // **** Reduction at tree node ****
      if ((min == -1 || max == -1) || (max - min > 50000)) {
        const unsigned accessor_idx =
            tree_ref->v_acc_[cur->node_type.tree.idx_mid];
        const float dist =
            functor(tree_ref->in_data_ref_[accessor_idx], my_task_.second);
        result_set->Insert(dist);
        // **********************************

        // Determine which child node to traverse next
        const auto axis = cur->node_type.tree.axis;
        const auto train = tree_ref->in_data_ref_[accessor_idx].data[axis];
        const auto dir = my_task_.second.data[axis] < train ? kdt::Dir::kLeft
                                                            : kdt::Dir::kRight;

        // Will update 'k_dist' (dependency)
        TraversalRecursive(cur->GetChild(dir));

        // Check if we need to traverse the other side (optional)
        if (const auto diff = functor(my_task_.second.data[axis], train);
            diff < result_set->WorstDist()) {
          TraversalRecursive(cur->GetChild(FlipDir(dir)));
        }
      } else {
        for (int i = min; i < max; i++) {
          const unsigned accessor_idx = tree_ref->v_acc_[i];
          const float dist =
              functor(tree_ref->in_data_ref_[accessor_idx], my_task_.second);
          result_set->Insert(dist);
        }
      }
    }
  }

 public:
  // Current processing task and its result (kSet)
  Task my_task_;

  union {
    float* my_assigned_result_addr;
    KnnSet<float, 1>* result_set = nullptr;
  };

  // Couroutine related
  std::vector<CallStackField> stack_;
  kdt::Node* cur_;
  ExecutionState state_;

  // Store some reference used (const)
  int my_tid_;
  int my_stream_id_;
  int my_uid_;
};
