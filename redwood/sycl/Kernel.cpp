#include "../Kernel.hpp"

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

constexpr auto kNumStreams = 2;

sycl::device device;
sycl::context ctx;
sycl::queue qs[kNumStreams];

// Global variable
const Point4F* usm_leaf_node_table = nullptr;

void ShowDevice(const sycl::queue& q) {
  // Output platform and device information.
  const auto device = q.get_device();
  const auto p_name =
      device.get_platform().get_info<sycl::info::platform::name>();
  std::cout << std::setw(20) << "Platform Name: " << p_name << "\n";
  const auto p_version =
      device.get_platform().get_info<sycl::info::platform::version>();
  std::cout << std::setw(20) << "Platform Version: " << p_version << "\n";
  const auto d_name = device.get_info<sycl::info::device::name>();
  std::cout << std::setw(20) << "Device Name: " << d_name << "\n";
  const auto max_work_group =
      device.get_info<sycl::info::device::max_work_group_size>();
  std::cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
  const auto max_compute_units =
      device.get_info<sycl::info::device::max_compute_units>();
  std::cout << std::setw(20) << "Max Compute Units: " << max_compute_units
            << "\n\n";
}

void SyclWarmUp(sycl::queue& q) {
  int sum;
  sycl::buffer<int> sum_buf(&sum, 1);
  q.submit([&](auto& h) {
    sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
    h.parallel_for(1, [=](auto) { sum_acc[0] = 0; });
  });
  q.wait();
}

struct MyFunctor {
  inline float operator()(const Point4F p, const Point4F q) const {
    auto dist = float();

    for (int i = 0; i < 4; ++i) {
      const auto diff = p.data[i] - q.data[i];
      dist += diff * diff;
    }

    return sqrtf(dist);
  }
};

namespace redwood::internal {

void BackendInitialization() {
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

void RegisterLeafNodeTable(const void* leaf_node_table,
                           const int num_leaf_nodes) {
  usm_leaf_node_table = static_cast<const Point4F*>(leaf_node_table);
}

// CUDA Only, Ignore it in SYCL.
void AttachStreamMem(const int stream_id, void* addr) {}

void DeviceSynchronize() {
  for (int i = 0; i < kNumStreams; ++i) qs[i].wait();
}

void DeviceStreamSynchronize(const int stream_id) { qs[stream_id].wait(); }

}  // namespace redwood::internal