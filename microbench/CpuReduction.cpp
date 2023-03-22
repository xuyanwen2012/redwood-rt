#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "DistanceMetrics.hpp"
#include "Point.hpp"

void clear_cpu_cache() {
  constexpr size_t cache_size = 10 * 1024 * 1024;  // 10 MB
  std::vector<char> cache_clearer(cache_size);
  for (auto& e : cache_clearer) {
    e = static_cast<char>(std::rand() % 256);
  }
  const volatile char result = std::accumulate(
      cache_clearer.begin(), cache_clearer.end(), static_cast<char>(0));
  // std::cout << "Cache cleared: " << static_cast<int>(result) << std::endl;
}

template <typename DistanceFunctor>
void benchmark(const std::vector<Float4>& data_points,
               const std::vector<Float4>& query_points,
               const DistanceFunctor& distance_functor,
               const std::string& name) {
  using namespace std::chrono;

  const auto start = high_resolution_clock::now();

  volatile float sum = 0.0f;

  for (const auto& query : query_points) {
    for (const auto& data : data_points) {
      const float distance = distance_functor(query, data);
      sum += distance;
    }
  }

  const auto end = high_resolution_clock::now();
  const auto duration = duration_cast<milliseconds>(end - start).count();
  const double throughput =
      (static_cast<double>(data_points.size()) * query_points.size()) /
      duration;

  std::cout << name << " Distance - Throughput: " << throughput
            << " distances/ms" << std::endl;
}

int main() {
  constexpr size_t data_points_count = 1024 * 1024;
  constexpr size_t query_points_count = 1024;

  const std::vector<Float4> data_points =
      GenerateRandomFloat4(data_points_count);
  const std::vector<Float4> query_points =
      GenerateRandomFloat4(query_points_count);

  const dist::Euclidean euclidean_distance;
  const dist::Manhattan manhattan_distance;
  const dist::Chebyshev chebyshev_distance;
  const dist::EuclideanSquared euclidean_distance_squared;
  const dist::Minkowski minkowski_distance(3);
  const dist::Canberra canberra_distance;
  const dist::BrayCurtis bray_curtis_distance;
  const dist::Cosine cosine_distance;

  using BenchmarkFunction = std::function<void()>;
  std::array<BenchmarkFunction, 8> benchmarks = {
      [&] {
        benchmark(data_points, query_points, euclidean_distance, "Euclidean");
      },
      [&] {
        benchmark(data_points, query_points, euclidean_distance_squared,
                  "Euclidean Squared");
      },
      [&] {
        benchmark(data_points, query_points, manhattan_distance, "Manhattan");
      },
      [&] {
        benchmark(data_points, query_points, chebyshev_distance, "Chebyshev");
      },
      [&] {
        benchmark(data_points, query_points, minkowski_distance, "Minkowski");
      },
      [&] {
        benchmark(data_points, query_points, canberra_distance, "Canberra");
      },
      [&] {
        benchmark(data_points, query_points, bray_curtis_distance,
                  "Bray Curtis");
      },
      [&] { benchmark(data_points, query_points, cosine_distance, "Cosine"); },
  };

  for (const auto& bm : benchmarks) {
    clear_cpu_cache();
    bm();
  }
}