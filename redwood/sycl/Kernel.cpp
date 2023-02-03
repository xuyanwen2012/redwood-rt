#include "../Kernel.hpp"

// #include <limits>
// #include "SyclUtils.hpp"

namespace redwood::internal {

void DeviceWarmUp() {
  // SyclWarmUp();
}

void RegisterLeafNodeTable(const void* leaf_node_table,
                           const int num_leaf_nodes) {}

// CUDA Only
void AttachStreamMem(const int stream_id, void* addr) {}

// Main entry to the NN Kernel
void ProcessNnBuffer(const Point4F* query_points, const int* query_idx,
                     const int* leaf_idx, const Point4F* leaf_node_table,
                     float* out, const int num, const int stream_id) {}

void DeviceSynchronize() {}

void DeviceStreamSynchronize(const int stream_id) {}

}  // namespace redwood::internal