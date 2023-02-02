#pragma once

#include <vector>

namespace redwood {

// Store:
// 1) host side reference to query point (Q set), which is used such as
// 'ReduceLeafNode(leaf_id)'
// 2) host side leaf_node_table_ref (Leaf nodes)
struct SharedData {
  //   const float *host_leaf_node_table_ref;
  const float* host_query_point_ref;

  // num_leaf_nodes * leaf_size
  // const float* usm_leaf_node_table;
};

// extern

}  // namespace redwood