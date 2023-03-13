#include "Redwood/Core.hpp"

#include <fcntl.h>
#include <sys/mman.h>

#include <cassert>
#include <iostream>

#include "DuetConsts.hpp"

// Main entry for Duet
volatile uint64_t* duet_baseaddr = nullptr;

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
