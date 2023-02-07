// #include <CL/sycl.hpp>
// #include <cmath>
// #include <cstddef>
// #include <iostream>
// #include <mutex>

// #include "../../include/PointCloud.hpp"
// #include "../../include/UsmAlloc.hpp"
// #include "../Kernel.hpp"

// constexpr auto kNumStreams = 2;
// constexpr auto kBlockThreads = 256;

// extern sycl::device device;
// extern sycl::context ctx;
// // TODO: this thing is needed per thread
// extern sycl::queue qs[kNumStreams];

// std::once_flag flag1;

// // SYCL workaround
// constexpr std::size_t intermediate_size = 128;
// Point3F* usm_intermediate_results[kNumStreams];
// int intermediate_count[kNumStreams];

// template <typename T>
// T MyRoundUp(T num_to_round, T multiple = 32) {
//   T remainder = num_to_round % multiple;
//   if (remainder == 0) return num_to_round;

//   return num_to_round + multiple - remainder;
// }

// inline auto rsqrtf(const float x) { return 1.0f / sqrtf(x); }

// inline Point3F KernelFunc(const Point4F p, const Point3F q) {
//   const auto dx = p.data[0] - q.data[0];
//   const auto dy = p.data[1] - q.data[1];
//   const auto dz = p.data[2] - q.data[2];
//   const auto dist_sqr = dx * dx + dy * dy + dz * dz + 1e-9f;
//   const auto inv_dist = rsqrtf(dist_sqr);
//   const auto inv_dist3 = inv_dist * inv_dist * inv_dist;
//   const auto with_mass = inv_dist3 * p.data[3];
//   return {dx * with_mass, dy * with_mass, dz * with_mass};
// }

// namespace redwood::internal {

// // Note, in SYCL implementation, do not use out directly.
// // Use intermediate result first.
// void ProcessBhBuffer(const Point3F query_point, const Point4F* leaf_node_table,
//                      const int* leaf_idx, const int num_leaf_collected,
//                      const Point4F* branch_data, const int num_branch_collected,
//                      Point3F* out, const int leaf_max_size,
//                      const int stream_id) {
//   std::call_once(flag1, [&] {
//     std::cout << "Initialize Intermediate Result for first time" << std::endl;

//     for (int i = 0; i < kNumStreams; ++i)
//       usm_intermediate_results[i] = static_cast<Point3F*>(
//           internal::UsmAlloc(intermediate_size * sizeof(Point3F)));
//   });

//   const auto data_size = num_leaf_collected;

//   if (data_size == 0) {
//     return;
//   }

//   // Each work is a leaf, 'data_size' == leaf collected in the pack
//   const auto num_work_items = MyRoundUp(data_size, kBlockThreads);
//   const auto num_work_groups = num_work_items / kBlockThreads;

//   std::cout << "num_work_items: " << num_work_items << std::endl;
//   std::cout << "num_work_groups: " << num_work_groups << std::endl;

//   if (num_work_groups > 1024) {
//     std::cout << "should not happen" << std::endl;
//     exit(1);
//   }

//   // Remember how many SYCL work gourps was uses, so later we can reduce them on
//   // the host once the results are produced
//   // pack.tmp_count_le = num_work_groups;
//   intermediate_count[stream_id] = num_work_groups;
//   const auto tmp_result_ptr = usm_intermediate_results[stream_id];

//   qs[stream_id].submit([&](sycl::handler& h) {
//     const sycl::local_accessor<Point3F, 1> scratch(kBlockThreads, h);

//     h.parallel_for(sycl::nd_range<1>(num_work_items, kBlockThreads),
//                    [=](const sycl::nd_item<1> item) {
//                      const auto global_id = item.get_global_id(0);
//                      const auto local_id = item.get_local_id(0);
//                      const auto group_id = item.get_group(0);

//                      if (global_id < data_size) {
//                        const auto leaf_id = leaf_idx[global_id];
//                        //  const auto leaf_size = leaf_sizes_acc[leaf_id];

//                        Point3F my_sum{};
//                        for (int i = 0; i < leaf_max_size; ++i) {
//                          my_sum += KernelFunc(
//                              leaf_node_table[leaf_id * leaf_max_size + i],
//                              query_point);
//                        }
//                        scratch[local_id] = my_sum;
//                      } else {
//                        scratch[local_id] = Point3F();
//                      }

//                      // Do a tree reduction on items in work-group
//                      for (int i = kBlockThreads / 2; i > 0; i >>= 1) {
//                        item.barrier(sycl::access::fence_space::local_space);
//                        if (local_id < i)
//                          scratch[local_id] += scratch[local_id + i];
//                      }

//                      if (local_id == 0) tmp_result_ptr[group_id] = scratch[0];
//                    });
//   });
// }

// void OnBhBufferFinish(Point3F* result, const int stream_id) {
//   const auto count = intermediate_count[stream_id];
//   auto local = Point3F();
//   for (int i = 0; i < count; ++i) {
//     local += usm_intermediate_results[stream_id][i];
//   }
//   *result += local;
// }

// }  // namespace redwood::internal