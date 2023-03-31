#include "Redwood/Core.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cassert>
#include <iostream>
#include <vector>

#include "Redwood/Duet/Consts.hpp"
#include "Redwood/Point.hpp"

// Main entry for Duet
// Makesure this is the only decleration
volatile uint64_t* duet_baseaddr = nullptr;

std::vector<int> reduction_called_counter;

namespace redwood {

void Init() {
  std::cout << "redwood::Init()" << std::endl;

  const unsigned leaf_size = 64;
  const unsigned num_threads = 1;

  assert(leaf_size % duet::kDuetLeafSize == 0);

  // 8k?
  int fd = open("/dev/duet", O_RDWR);

  if (fd == -1) {
    duet_baseaddr =
        static_cast<volatile uint64_t*>(malloc(duet::kNEngine << 13));

    std::cout << "[info] Failed to open \'/dev/duet\', using malloc instead!"
              << std::endl;

  } else {
    // Only useful when we are simulating in Gem5
    duet_baseaddr = static_cast<volatile uint64_t*>(
        mmap(nullptr, duet::kNEngine << 13, PROT_READ | PROT_WRITE, MAP_PRIVATE,
             fd, 0));
  }

  for (long i = 0; i < duet::kNEngine; ++i) {
    duet_baseaddr[i << 10] = static_cast<uint64_t>(duet::kEpssq);
  }

  std::cout << "[info] Duet engine has been initialized!" << std::endl;
}

void DeviceSynchronize() {}

void DeviceStreamSynchronize(int stream_id) {}

void AttachStreamMem(int stream_id, void* addr) {}

}  // namespace redwood
