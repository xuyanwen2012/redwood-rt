#pragma once

#include "Constants.hpp"
#include "Macros.hpp"

namespace redwood {

// --- Public APIs (need to be implemented by the backends) --
void Init();

void DeviceSynchronize();
void DeviceStreamSynchronize(int stream_id);

void AttachStreamMem(int stream_id, void* addr);

// CUDA Only
inline void AttachStream(int stream_id, void* addr) {
  if constexpr (kRedwoodBackend == redwood::Backends::kCuda)
    AttachStreamMem(stream_id, addr);
}

}  // namespace redwood