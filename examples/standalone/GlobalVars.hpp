#pragma once

#include <memory>
#include <vector>

#include "KDTree.hpp"

// Global vars
inline std::shared_ptr<kdt::KdTree> tree_ref;

// Debug
inline std::vector<std::vector<int>> leaf_node_visited1;
inline std::vector<std::vector<int>> leaf_node_visited2;
inline std::vector<float> final_results1;
inline std::vector<float> final_results2;
