#pragma once

#include <iostream>

// For NN and KNN
struct AppParams {
  int max_leaf_size;
  int batch_size;
  int num_threads;
  int m;
  bool cpu;
  bool dump;
};

inline std::ostream& operator<<(std::ostream& os, const AppParams& params) {
  os << "Application Parameters:\n";
  os << "\tMax Leaf Size: " << params.max_leaf_size << '\n';
  os << "\tBatch Size: " << params.batch_size << '\n';
  os << "\tNum Threads: " << params.num_threads << '\n';
  os << "\tM: " << params.m << '\n';
  os << "\tRunning Cpu: " << std::boolalpha << params.cpu << '\n';
  os << "\tDump file: " << std::boolalpha << params.dump << '\n';
  return os;
}

inline AppParams app_params;
