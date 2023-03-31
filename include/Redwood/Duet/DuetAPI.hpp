#pragma once

namespace duet {

// Set anchor registers
// Like 'x0, y0, z0'
void Start(const long tid, const void* query_element);

// Push leaf_no_base_addr to the duet registers
void PushLeaf32(const long tid, const void* node_base_addr);

// Simple reduction on host
// void PushBranch(long tid, const void* node_element, unsigned query_idx);

}  // namespace duet