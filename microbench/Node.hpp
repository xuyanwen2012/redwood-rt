#pragma once

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

