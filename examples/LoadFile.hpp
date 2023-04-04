#pragma once

#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "Utils.hpp"

template <typename T>
_NODISCARD std::vector<T> load_data_from_file(const std::string& filename) {
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

template <typename T>
void DumpFile(const std::vector<T>& data, const bool is_cpu) {
  std::ofstream outfile(is_cpu ? "output_c.txt" : "output.txt");

  if (outfile.is_open()) {
    for (const auto& element : data) {
      outfile << element << '\n';
    }
    outfile.close();
  }
}

template <typename T>
_NODISCARD std::vector<T> read_floats_from_file(const std::string& filename,
                                                const int n, const int m) {
  std::vector<T> in_data(n);
  std::vector<T> q_data(m);

  std::ifstream infile(filename, std::ios::binary);

  infile.seekg(0, std::ios::end);
  std::streamsize size = infile.tellg();
  infile.seekg(0, std::ios::beg);

  if (n + m > size / sizeof(T)) {
    throw std::runtime_error("Not enough data in the file: " + filename);
  }

  if (infile) {
    infile.read(reinterpret_cast<char*>(in_data.data()), n * sizeof(T));
    infile.read(reinterpret_cast<char*>(q_data.data()), m * sizeof(T));
    infile.close();
  } else {
    std::cerr << "Error opening file" << std::endl;
  }

  return in_data;
}
