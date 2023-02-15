#pragma once

#include "Constants.hpp"

namespace redwood {

// --- Public APIs (need to be implemented by the backends) --
void Init();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

// CUDA Only
void AttachStreamMem(int stream_id, void* addr);

}  // namespace redwood