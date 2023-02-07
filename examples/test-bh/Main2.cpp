// #include <iostream>
// #include <random>
// #include <vector>

// // Redwood related
// #include "../../redwood/BhBuffer.hpp"
// #include "../../redwood/Kernel.hpp"
// #include "PointCloud.hpp"
// #include "Redwood.hpp"
// #include "UsmAlloc.hpp"

// float my_rand(float min = 0.0, float max = 1.0) {
//   // 114514 and 233
//   static thread_local std::mt19937 generator(114514);  // NOLINT(cert-msc51-cpp)
//   std::uniform_real_distribution<float> distribution(min, max);
//   return distribution(generator);
// }

// namespace redwood {
// namespace internal {
// extern void ProcessBhBuffer(const Point3F query_point,
//                             const Point4F* leaf_node_table, const int* leaf_idx,
//                             int num_leaf_collected, const Point4F* branch_data,
//                             int num_branch_collected, Point3F* out,
//                             int leaf_max_size, int stream_id);
// }
// }  // namespace redwood

// int main() {
//   // const auto n = 1024 * 32;
//   const auto m = 1024;
//   const auto batch_size = 1024;
//   const auto leaf_size = 32;

//   // auto in_data = static_cast<Point4F*>(malloc(n * sizeof(Point4F)));

//   redwood::UsmVector<Point4F> leaf_node_table(1024 * leaf_size);

//   auto q_data = static_cast<Point3F*>(malloc(m * sizeof(Point3F)));

//   static auto rand_point4f = []() {
//     return Point4F{
//         my_rand(0.0f, 1000.0f),
//         my_rand(0.0f, 1000.0f),
//         my_rand(0.0f, 1000.0f),
//         my_rand(0.0f, 1000.0f),
//     };
//   };

//   static auto rand_point3f = []() {
//     return Point3F{
//         my_rand(0.0f, 1000.0f),
//         my_rand(0.0f, 1000.0f),
//         my_rand(0.0f, 1000.0f),
//     };
//   };

//   std::generate(leaf_node_table.begin(), leaf_node_table.end(), rand_point4f);
//   std::generate_n(q_data, m, rand_point3f);

//   redwood::BhBuffer<Point4F, Point3F, Point3F> bh_pack;

//   bh_pack.Allocate(batch_size);

//   const auto q_idx = 2;
//   bh_pack.SetTask(q_data[q_idx], q_idx);

//   for (int i = 0; i < 512; ++i) {
//     bh_pack.PushLeaf(i);
//   }

//   for (int i = 0; i < 256; ++i) {
//     bh_pack.PushBranch(rand_point4f());
//   }

//   redwood::UsmVector<Point3F> final_result(m);

//   redwood::InitReducer(1, leaf_size, 1024, 1024);

//   const auto tid = 0;

//   redwood::SetQueryPoints(tid, q_data, m);
//   redwood::SetNodeTables(leaf_node_table.data(), 1024);

//   redwood::internal::ProcessBhBuffer(
//       bh_pack.my_query, leaf_node_table.data(), bh_pack.LeafNodeData(),
//       bh_pack.NumLeafsCollected(), nullptr, 0, nullptr, leaf_size, 0);

//   redwood::internal::DeviceStreamSynchronize(0);

//   redwood::internal::OnBhBufferFinish(final_result.data() + q_idx, 0);

//   for (int i = 0; i < 32; ++i) {
//     std::cout << i << ": " << final_result[i] << std::endl;
//   }

//   return EXIT_SUCCESS;
// }