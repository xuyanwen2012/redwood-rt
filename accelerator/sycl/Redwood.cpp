#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>


#include "SyclUtils.hpp"

#include "Redwood/Point.hpp"

sycl::device device;
sycl::context ctx;
sycl::queue qs[kNumStreams];


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

void DeviceStreamSynchronize(const int stream_id) {
    qs[stream_id].wait();
}

void DeviceSynchronize() {
for (int i = 0; i < kNumStreams; ++i) DeviceStreamSynchronize(i);
 }

void AttachStreamMem(const int stream_id, void* addr) {
    // No Op
}

void* UsmMalloc(std::size_t n) {
  void* tmp;
  std::cout << "accelerator::UsmMalloc() " << tmp << ": " << n << " bytes."
            << std::endl;
  tmp = sycl::malloc_shared(n, device, ctx);
  return tmp;
}

void UsmFree(void* ptr) {
  std::cout << "accelerator::UsmFree() " << ptr << std::endl;
  if (ptr) {
    sycl::free(ptr, ctx);
  }
}

void ComputeOneBatchAsync(const int* u_leaf_indices,  /**/
                          const int num_active_leafs, /**/
                          float* out,                 /**/
                          const Point4F* u_lnt_data,  /**/
                          const int* u_lnt_sizes,     /**/
                          const Point4F q,            /**/
                          const int stream_id) {
  
}

void ProcessKnnAsync(const int* u_leaf_indices,  /**/
                     const Point4F* u_q_points,  /**/
                     const int num_active_leafs, /**/
                     float* out,                 /**/
                     const Point4F* u_lnt_data,  /**/
                     const int* u_lnt_sizes,     /**/
                     const int stream_id) {

}

}