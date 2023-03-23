#pragma once

#include <fstream>
#include <iostream>
#include <vector>

template <typename T>
std::vector<T> load_data_from_file(const std::string& filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Read all floats from the input file
  infile.seekg(0, std::ios::end);
  std::streamsize size = infile.tellg();
  infile.seekg(0, std::ios::beg);

  std::vector<T> data(size / sizeof(T));
  infile.read(reinterpret_cast<char*>(data.data()), size);

  if (infile.gcount() != size) {
    throw std::runtime_error("Failed to read entire file: " + filename);
  }

  return data;
}