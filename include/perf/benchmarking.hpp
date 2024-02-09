// Using chrono for good measurements and parallelism support

#pragma once
#include <omp.h>

#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

template <typename DurationContainerType, typename DurationType, class Function,
          typename... Args>
DurationContainerType elapsed_time(Function&& func, Args&&... args) {
  auto duration = std::chrono::duration<DurationContainerType, DurationType>();
  auto start = std::chrono::high_resolution_clock::now();
  func(args...);
  auto end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  return duration.count();
}

// returns time in seconds
template <class Function, typename... Args>
double elapsed_time_omp(Function&& func, Args&&... args) {
  double start = omp_get_wtime();
  func(args...);
  double end = omp_get_wtime();
  return end - start;
}

template <typename DurationContainerType, typename DurationType, class Function,
          typename... Args>
DurationContainerType elapsed_time_avg(const size_t iters, Function&& func,
                                       Args&&... args) {
  auto duration = std::chrono::duration<DurationContainerType, DurationType>();
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < iters; i++) {
    func(args...);
  }
  auto end = std::chrono::high_resolution_clock::now();
  duration = (end - start) / iters;
  return duration.count();
}

// returns time in seconds
template <class Function, typename... Args>
double elapsed_time_omp_avg(const size_t iters, Function&& func,
                            Args&&... args) {
  double start = omp_get_wtime();
  for (size_t i = 0; i < iters; i++) {
    func(args...);
  }
  double end = omp_get_wtime();
  return (end - start) / iters;
}

template <typename ThroughputContainerType, typename DurationType,
          class Function, typename... Args>
ThroughputContainerType throughput(Function&& func, Args&&... args) {
  return ThroughputContainerType(1) /
         elapsed_time<ThroughputContainerType, DurationType>(func, args...);
}

template <class Function, typename... Args>
double throughput_omp(Function&& func, Args&&... args) {
  return 1 / elapsed_time_omp(func, args...);
}

template <typename ThroughputContainerType, typename DurationType,
          class Function, typename... Args>
ThroughputContainerType throughput_avg(const size_t iters, Function&& func,
                                       Args&&... args) {
  return ThroughputContainerType(1) /
         elapsed_time_avg<ThroughputContainerType, DurationType>(iters, func,
                                                                 args...);
}

template <class Function, typename... Args>
double throughput_omp_avg(const size_t iters, Function&& func, Args&&... args) {
  return 1 / elapsed_time_omp_avg(iters, func, args...);
}

// as "Manhattan" norm of error-vector
template <typename T>
T accuracy(T* test, T* ref, const size_t size) {
  if (test == nullptr || ref == nullptr) {
    throw std::invalid_argument("Bad pointer");
  }
  std::vector<T> differ(size);
  for (size_t i = 0; i < size; i++) {
    differ[i] = std::abs(test[i] - ref[i]);
  }
  // sum of differences
  T res = std::accumulate(differ.begin(), differ.end(), T(0));
  return res;
}

// as Euclidean norm of error-vector
// assume that (T)*(T)>=T(0); (T)*(T)=T(0) <=> multiplying 0s
template <typename T>
T accuracy_norm(T* test, T* ref, const size_t size) {
  if (test == nullptr || ref == nullptr) {
    throw std::invalid_argument("Bad pointer");
  }
  std::vector<T> differ(size);
  for (size_t i = 0; i < size; i++) {
    differ[i] = (test[i] - ref[i]) * (test[i] - ref[i]);
  }
  // sum of squares
  T res = std::accumulate(differ.begin(), differ.end(), T(0));
  // typename T should have friend sqrt() function
  return std::sqrt(res);
}
