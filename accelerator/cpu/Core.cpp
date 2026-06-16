#include "Redwood/Core.hpp"

namespace redwood {

// The CPU backend executes every "accelerator" kernel synchronously on the
// host inside NearestNeighborKernel(). There is therefore no asynchronous work
// queue to drain and no device memory to attach: the synchronization and
// stream-attach entry points are intentionally no-ops. They exist so that the
// same application/runtime code that targets CUDA or SYCL also links and runs
// unchanged against this reference backend.

int stored_num_threads = 1;

void Init(int num_threads) { stored_num_threads = num_threads; }

void DeviceSynchronize() {}

void DeviceStreamSynchronize(int /*tid*/, int /*stream_id*/) {}

void AttachStreamMem(int /*tid*/, int /*stream_id*/, void* /*addr*/) {}

}  // namespace redwood
