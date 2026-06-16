#include "Redwood/Core.hpp"

#include <sycl/sycl.hpp>

#include <iostream>

#include "Consts.hpp"

// Backend-wide SYCL state, shared with Usm.cpp and Kernel.cpp via extern.
// The context must be created and stored here (the previous version left the
// global `ctx` default-constructed, so USM allocations used the wrong context).
sycl::device g_device;
sycl::context g_context;
sycl::queue g_queues[kNumStreams];

namespace redwood {

void Init(int /*num_threads*/) {
  // default_selector_v picks the best available device. In CI / containers
  // that is typically the OpenCL CPU runtime; on a SYCL-capable GPU it picks
  // the GPU. (The old code hard-coded gpu_selector_v, which aborts when no GPU
  // is present.)
  g_device = sycl::device(sycl::default_selector_v);
  g_context = sycl::context(g_device);
  for (int i = 0; i < kNumStreams; ++i) {
    g_queues[i] = sycl::queue(g_context, g_device);
  }

  std::cout << "redwood(SYCL)::Init device: "
            << g_device.get_info<sycl::info::device::name>() << std::endl;
}

void DeviceStreamSynchronize(int /*tid*/, int stream_id) {
  g_queues[stream_id].wait();
}

void DeviceSynchronize() {
  for (auto& q : g_queues) q.wait();
}

void AttachStreamMem(int /*tid*/, int /*stream_id*/, void* /*addr*/) {
  // No-op: malloc_shared memory is accessible from every queue in the context.
}

}  // namespace redwood
