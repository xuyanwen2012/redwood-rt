#include <fcntl.h>
#include <sys/mman.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "Consts.hpp"
#include "Redwood/Core.hpp"
#include "Redwood/Point.hpp"

constexpr auto kDebugPrint = false;

// Main entry for Duet
volatile uint64_t* duet_baseaddr = nullptr;

std::vector<int> reduction_called_counter;

namespace redwood {

void Init() {
  const unsigned leaf_size = 64;
  const unsigned num_threads = 1;

  assert(leaf_size % kDuetLeafSize == 0);

  // 8k?
  int fd = open("/dev/duet", O_RDWR);
  duet_baseaddr = static_cast<volatile uint64_t*>(mmap(
      nullptr, kNEngine << 13, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0));

  for (long i = 0; i < kNEngine; ++i) {
    duet_baseaddr[i << 10] = static_cast<const uint64_t>(kEpssq);
  }
}

void DeviceSynchronize() {}

void DeviceStreamSynchronize(int stream_id) {}

void AttachStreamMem(int stream_id, void* addr) {}

}  // namespace redwood

void StartQuery(const long tid, const void* query_element) {
  auto ptr = reinterpret_cast<const Point3D*>(query_element);

  const long caller_id = tid;
  volatile uint64_t* sri = duet_baseaddr + (caller_id << 4) + 16;

  if constexpr (kDebugPrint) {
    std::cout << tid << ": started duet. " << ptr->data[0] << std::endl;
  }

  sri[kPos0X] = *reinterpret_cast<const uint64_t*>(&ptr->data[0]);
  sri[kPos0Y] = *reinterpret_cast<const uint64_t*>(&ptr->data[1]);
  sri[kPos0Z] = *reinterpret_cast<const uint64_t*>(&ptr->data[2]);
}

inline void ReduceLeafNode(const long tid, const void* node_base_addr) {
  const long caller_id = tid;
  volatile uint64_t* sri = duet_baseaddr + (caller_id << 4) + 16;

  if constexpr (kDebugPrint) {
    auto ptr = reinterpret_cast<const Point4D*>(node_base_addr);
    std::cout << tid << ": pushed duet. " << ptr->data[0]
              << "\taddress: " << node_base_addr << std::endl;
  }

  sri[kArg] = reinterpret_cast<uint64_t>(node_base_addr);
}

void ReduceLeafNode(const long tid, const unsigned node_idx,
                    const unsigned query_idx) {
  // Must set 'leaf_nodes_data' before calling this function
  if constexpr (kDebugPrint) {
    std::cout << tid << ": ReduceLeafNode, node id:  " << node_idx << std::endl;
  }

  // const auto next_32 = MyRoundUp<int>(leaf_sizes[node_idx]);
  // for (int i = 0; i < next_32; i += kDuetLeafSize) {
  //   // Each call takes the next 'kDuetLeafSize' elements from that base
  //   pointer. ReduceLeafNode(tid, &leaf_nodes_data[node_idx * stored_leaf_size
  //   + i]);
  //   ++reduction_called_counter[query_idx];
  // }
}

void ReduceBranchNode(long tid, const void* node_element, unsigned query_idx) {
  // auto kernel_func = MyFunctor();
  // const auto p = static_cast<const Point4D*>(node_element);

  // branch_results[query_idx] += kernel_func(*p, query_data_base[query_idx]);
}
