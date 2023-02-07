#pragma once

namespace redwood::accelerator {

void Initialization();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

// CUDA Only
void AttachStreamMem(int stream_id, void* addr);

}  // namespace redwood::accelerator