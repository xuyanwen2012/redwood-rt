#pragma once

#include <vector>

namespace redwood {

// --- Public APIs (need to be implemented by the backends) --
void Init(int num_threads);

void DeviceSynchronize();
void DeviceStreamSynchronize(int tid, int stream_id);

// only useful in CUDA
void AttachStreamMem(int tid, int stream_id, void* addr);

}  // namespace redwood