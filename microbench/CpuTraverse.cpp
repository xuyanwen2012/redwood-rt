#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;
using namespace std::chrono;

// Define the tree node struct
struct Node {
  int val;
  Node* left;
  Node* right;

  Node(int value) {
    val = value;
    left = nullptr;
    right = nullptr;
  }
};

// Define the sparse tree class
class SparseTree {
 private:
  Node* root;
  vector<Node*> nodes;

 public:
  SparseTree() { root = nullptr; }

  ~SparseTree() {
    for (Node* node : nodes) {
      delete node;
    }
  }

  void insert(int value) {
    Node* node = new Node(value);
    nodes.push_back(node);

    if (root == nullptr) {
      root = node;
      return;
    }

    Node* current = root;
    while (current != nullptr) {
      if (value < current->val) {
        if (current->left == nullptr) {
          current->left = node;
          return;
        }
        current = current->left;
      } else {
        if (current->right == nullptr) {
          current->right = node;
          return;
        }
        current = current->right;
      }
    }
  }

  void randomTraversal() {
    if (root == nullptr) {
      return;
    }

    Node* current = root;
    while (current != nullptr) {
      if (rand() % 2 == 0) {
        current = current->left;
      } else {
        current = current->right;
      }
    }
  }
};

int main() {
  SparseTree tree;

  // Insert 1 million random values into the tree
  for (int i = 0; i < 1000000; i++) {
    int value = rand();
    tree.insert(value);
  }

  // Benchmark random traversal throughput
  auto start = high_resolution_clock::now();

  int nodesVisited = 0;
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(stop - start);
  while (duration.count() < 1000) {
    tree.randomTraversal();
    nodesVisited++;
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
  }

  double throughput = static_cast<double>(nodesVisited) / duration.count();
  cout << "Throughput: " << throughput << " nodes/ms" << endl;

  return EXIT_SUCCESS;
}
