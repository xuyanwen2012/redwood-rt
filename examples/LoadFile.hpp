#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>

std::pair<float *, size_t> LoadData(const std::string &filename) {
  struct stat s;
  stat(filename.c_str(), &s);

  int fd = open(filename.c_str(), O_RDONLY, 0);
  if (fd == -1) {
    std::cerr << "Failed to open file. Abort." << std::endl;
    exit(1);
  }

  float *map = reinterpret_cast<float *>(
      mmap(nullptr, s.st_size, PROT_READ, MAP_SHARED, fd, 0));

  close(fd);

  const auto data_size = s.st_size / sizeof(float);

  return {map, data_size};
}