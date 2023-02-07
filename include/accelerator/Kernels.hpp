#pragma once

#include "../include/PointCloud.hpp"

namespace redwood::accelerator {

// TODO: these APIs are for demonstration only
// Need to provide implementations.

// Specific to NN (Naive version)
// Note: NN <Point4F, Point4F, float>
void LaunchNnKernel(const Point4F* query_points, const Point4F* leaf_node_table,
                    const int* query_idx, const int* leaf_idx, float* out,
                    int num, int leaf_max_size, int stream_id);

// Specific to BH (Naive version)
// Only process on a single query point, and 'out' is probabaly a single address
// Note: BH <Point4F, Point3F, Point3F>
void LaunchBhKernel(const Point3F query_point, const Point4F* leaf_node_table,
                    const int* leaf_idx, int num_leaf_collected,
                    const Point4F* branch_data, int num_branch_collected,
                    Point3F* out, int leaf_max_size, int stream_id);

}  // namespace redwood::accelerator