#pragma once

namespace duet {

void StartQuery(const long tid, const void* query_element);

void ReduceLeafNode32(const long tid, const void* node_base_addr);

void ReduceBranchNode(long tid, const void* node_element, unsigned query_idx);

}  // namespace duet