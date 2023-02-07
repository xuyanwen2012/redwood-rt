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
//   const auto m = 1024;

//   const auto batch_size = 1024;

//   const auto num_leaf_nodes = 1024;
//   const auto leaf_size = 32;

//   redwood::UsmVector<Point4F> leaf_node_table(num_leaf_nodes * leaf_size);

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

//   redwood::InitReducer(1, leaf_size, 1024, 1024);

//   const auto tid = 0;

//   redwood::SetQueryPoints(tid, q_data, m);
//   redwood::SetNodeTables(leaf_node_table.data(), num_leaf_nodes);

//   redwood::StartQuery(tid, 0);

//   for (int i = 0; i < 512; ++i) {
//     redwood::ReduceLeafNode(tid, i, 0);
//   }

//   redwood::rt::ExecuteCurrentBufferAsync(tid, 0);

//   redwood::StartQuery(tid, 1);

//   for (int i = 512; i < 512 + 256; ++i) {
//     redwood::ReduceLeafNode(tid, i, 1);
//   }

//   // redwood::rt::ExecuteCurrentBufferAsync(tid, 0);

//   redwood::EndReducer();

//   for (int i = 0; i < 32; ++i) {
//     Point3F a{666};
//     redwood::GetReductionResult(tid, i, &a);
//     std::cout << i << ": " << a << std::endl;
//   }

//   return EXIT_SUCCESS;
// }