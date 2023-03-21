#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stack>

#include "../../examples/Utils.hpp"

struct TreeNode {
  int val;
  TreeNode* left;
  TreeNode* right;

  explicit TreeNode(const int x) : val(x), left(), right() {}

  _NODISCARD bool IsLeaf() const { return left == nullptr && right == nullptr; }
};

TreeNode* insert(TreeNode* root, const int val) {
  if (root == nullptr) return new TreeNode(val);

  if (val < root->val)
    root->left = insert(root->left, val);
  else
    root->right = insert(root->right, val);

  return root;
}

void InorderTraversal(const TreeNode* cur) {
  if (cur == nullptr) return;

  if (cur->IsLeaf()) {
    std::cout << "leaf " << cur->val << std::endl;

  } else {
    std::cout << "branch pre " << cur->val << std::endl;

    InorderTraversal(cur->left);

    std::cout << "branch in " << cur->val << std::endl;

    InorderTraversal(cur->right);
  }
}

enum class ExecutionState { kWorking, kFinished };

class Executor {
  std::stack<TreeNode*> node_stack_;
  TreeNode* cur_;

  ExecutionState state_ = ExecutionState::kFinished;

 public:
  void StartTraversal(TreeNode* root) {
    cur_ = root;
    Execute();
  }

  void Resume() { Execute(); }

  _NODISCARD bool Finished() const {
    return state_ == ExecutionState::kFinished;
  }

 private:
  void Execute() {
    if (state_ == ExecutionState::kWorking) goto my_resume_point;
    state_ = ExecutionState::kWorking;

    while (cur_ != nullptr || !node_stack_.empty()) {
      while (cur_ != nullptr) {
        if (cur_->IsLeaf()) {
          // **** pre ****
          std::cout << "leaf " << cur_->val << std::endl;
          // *************

          // **** Coroutine return (API) ****
          return;
        my_resume_point:
          // ****************************

          cur_ = nullptr;
          continue;
        }

        // **** branch pre ****
        std::cout << "branch pre " << cur_->val << std::endl;
        // *******************

        node_stack_.push(cur_);
        cur_ = cur_->left;
      }

      if (!node_stack_.empty()) {
        cur_ = node_stack_.top();
        node_stack_.pop();

        // **** branch in ****
        std::cout << "branch in " << cur_->val << std::endl;
        // *******************

        cur_ = cur_->right;
      }
    }

    state_ = ExecutionState::kFinished;
  }
};

int main() {
  constexpr int n = 32;
  srand(666);

  TreeNode* root = nullptr;

  for (int i = 0; i < n; i++) {
    const int val = rand() % 1000;
    root = insert(root, val);
  }

  std::cout << "Inorder Traversal of the Binary Search Tree:\n";
  InorderTraversal(root);

  std::cout << "-----------------------" << std::endl;

  Executor exe{};

  exe.StartTraversal(root);

  while (!exe.Finished()) {
    exe.Resume();
  }

  return 0;
}
