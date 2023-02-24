#pragma once

namespace duet {

void Start(const long tid, const void* query_element);

void PushLeaf32(const long tid, const void* node_base_addr);

void PushBranch(long tid, const void* node_element, unsigned query_idx);

}  // namespace duet