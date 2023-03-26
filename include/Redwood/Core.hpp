#pragma once

#include <vector>

namespace redwood {

// --- Public APIs (need to be implemented by the backends) --
void Init();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

// only useful in CUDA
void AttachStreamMem(int stream_id, void* addr);

}  // namespace redwood