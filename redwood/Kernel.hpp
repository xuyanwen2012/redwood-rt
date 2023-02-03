#pragma once

#include "../include/PointCloud.hpp"

namespace redwood::internal {

void BackendInitialization();
void DeviceWarmUp();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

// CUDA Only
void AttachStreamMem(int stream_id, void* addr);

void RegisterLeafNodeTable(const void* leaf_node_table, int num_leaf_nodes);

// Specific to NN
void ProcessNnBuffer(const Point4F* query_points, const int* query_idx,
                     const int* leaf_idx, const Point4F* leaf_node_table,
                     float* out, int num, int stream_id);

}  // namespace redwood::internal