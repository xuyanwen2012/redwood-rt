#pragma once

namespace redwood {

void InitReducer(int num_threads = 1, int leaf_size = 32, int batch_num = 1024,
                 int batch_size = 1024);

// Mostly used by BH, or FPGAs
void StartQuery(int tid, int query_idx);

void ReduceLeafNode(int tid, int node_idx, int query_idx);
void ReduceLeafNode(int tid, int node_idx, int query_idx, const void* q);

void ReduceBranchNode(int tid, const void* node_element, int query_idx);

void GetReductionResult(int tid, int query_idx, void* result);
void EndReducer();

void SetQueryPoints(int tid, const void* query_points, int num_query);
void SetNodeTables(const void* usm_leaf_node_table, int num_leaf_nodes);

}  // namespace redwood