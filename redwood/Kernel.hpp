#pragma once

#include "../include/PointCloud.hpp"

namespace redwood::internal {

void BackendInitialization();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

void RegisterLeafNodeTable(const void* leaf_node_table, int num_leaf_nodes);

// CUDA Only
void AttachStreamMem(int stream_id, void* addr);

// Specific to NN (Naive version)
void ProcessNnBuffer(const void* query_points, const int* query_idx,
                     const int* leaf_idx, float* out, int num,
                     int leaf_max_size, int stream_id);

// Specific to BH (Naive version)
void ProcessBhBuffer(const void* query_points, const int query_idx,
                     const int* leaf_idx, const Point4F* branch_data,
                     Point3F* out, int num, int leaf_max_size, int stream_id);

}  // namespace redwood::internal