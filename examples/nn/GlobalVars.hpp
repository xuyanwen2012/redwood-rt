#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "KDTree.hpp"

// Global vars
inline std::shared_ptr<kdt::KdTree> tree_ref;

// Debug
// inline std::vector<std::vector<int>> leaf_node_visited1;
// inline std::vector<std::vector<int>> leaf_node_visited2;
inline std::vector<float> final_results1;
inline std::vector<float> final_results2;

inline void PrintLeafNodeVisited(const std::vector<std::vector<int>>& d,
                                 size_t n) {
  n = std::min(n, d.size());

  for (auto i = 0u; i < n; ++i) {
    std::cout << "Query " << i << ": [";
    for (const auto& elem : d[i]) {
      std::cout << elem;
      if (elem != d[i].back()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

inline void PrintFinalResult(const std::vector<float>& d, size_t n) {
  n = std::min(n, d.size());
  for (auto i = 0u; i < n; ++i)
    std::cout << "Query " << i << ": " << d[i] << '\n';
  std::cout << std::endl;
}
