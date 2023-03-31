
#include "Redwood/Core.hpp"

#include <CL/sycl.hpp>

#include "Consts.hpp"
#include "SyclUtils.hpp"

// Global Variables
sycl::device device;
sycl::context ctx;
sycl::queue qs[kNumStreams];

void SyclWarmUp(sycl::queue& q) {
  int sum;
  sycl::buffer<int> sum_buf(&sum, 1);
  q.submit([&](auto& h) {
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
  });
  q.wait();
}

namespace redwood {

void Init() {
  try {
    device = sycl::device(sycl::gpu_selector_v);
  } catch (const sycl::exception& e) {
    std::cout << "Cannot select a GPU\n" << e.what() << "\n";
    exit(1);
  }

  qs[0] = sycl::queue(device);
  for (int i = 1; i < kNumStreams; i++)
    qs[i] = sycl::queue(qs[0].get_context(), device);

  ShowDevice(qs[0]);
  SyclWarmUp(qs[0]);
}

void DeviceStreamSynchronize(const int stream_id) { qs[stream_id].wait(); }

void DeviceSynchronize() {
  for (int i = 0; i < kNumStreams; ++i) DeviceStreamSynchronize(i);
}

void AttachStreamMem(const int stream_id, void* addr) {
  // No Op
}

}  // namespace redwood