#pragma once

namespace redwood::internal {

void DeviceWarmUp();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

// CUDA Only
void AttachStreamMem(int stream_id, void* addr);

void RegisterLeafNodeTable(const void* leaf_node_table, int num_leaf_nodes);

// Specific to NN
void ProcessNnBuffer(const float* query_points, const int* query_idx,
                     const int* leaf_idx, const float* leaf_node_table,
                     float* out, int num, int stream_id);

}  // namespace redwood::internal