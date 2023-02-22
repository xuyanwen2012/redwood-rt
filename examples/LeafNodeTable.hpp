#pragma once

#include "Redwood/Point.hpp"
#include "Redwood/Usm.hpp"
#include "Utils.hpp"

namespace rdc {

// POD
inline struct LeafNodeTable {
  Point4F* u_data = nullptr;
  int* u_sizes = nullptr;
  int leaf_size;
  bool use_sizes;
} lnt;

inline void AllocateLeafNodeTable(const int num_leaf_nodes, const int leaf_size,
                                  const bool use_sizes = false) {
  lnt.u_data = redwood::UsmMalloc<Point4F>(num_leaf_nodes * leaf_size);

  if (use_sizes) lnt.u_sizes = redwood::UsmMalloc<int>(num_leaf_nodes);

  lnt.leaf_size = leaf_size;
  lnt.use_sizes = use_sizes;
}

inline void FreeLeafNodeTalbe() {
  redwood::UsmFree(lnt.u_data);
  if (lnt.use_sizes) redwood::UsmFree(lnt.u_sizes);
}

_NODISCARD inline Point4F* LntDataAddr() { return lnt.u_data; }

_NODISCARD inline const Point4F* LntDataAddrAt(const int node_idx) {
  return lnt.u_data + node_idx * lnt.leaf_size;
}

_NODISCARD inline int* LntSizesAddr() { return lnt.u_sizes; }

}  // namespace rdc