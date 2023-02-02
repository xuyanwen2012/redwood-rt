#pragma once

namespace redwood {

void InitReducer(int num_threads = 1, int leaf_size = 32, int batch_num = 1024,
                 int batch_size = 1024);

void StartQuery(int tid, int query_idx);
void StartQuery(int tid, const void* task_obj);

void ReduceLeafNode(int tid, int node_idx, int query_idx);
void ReduceBranchNode(int tid, const void* node_element, int query_idx);

void GetReductionResult(int tid, int query_idx, void* result);
void EndReducer();

void SetQueryPoints(int tid, const void* query_points, int num_query);
void SetNodeTables(const void* usm_leaf_node_table, int num_leaf_nodes);

namespace rt {

// Developer APIs
// Redwood developers can use the following APIs to micro controll the execution
// details. This particular function is used for GPU backend Executor Runtime.
void ExecuteCurrentBufferAsync(int tid, int num_batch_collected);

void ExecuteBufferAsync(int tid, int stream_id, int num_batch_collected);
void ExecuteBuffer(int tid, int stream_id, int num_batch_collected);
}  // namespace rt

}  // namespace redwood