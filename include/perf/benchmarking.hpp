// Using chrono for good measurements and parallelism support

#pragma once
#include <omp.h>

#include <chrono>
#include <cmath>
#include <stdexcept>

template <typename DurationContainerType, typename DurationType, class Function,
          typename... Args>
DurationContainerType elapsed_time(Function&& func, Args&&... args) {
  auto chronotimer = std::chrono::high_resolution_clock();
  auto duration = std::chrono::duration<DurationContainerType, DurationType>();
  auto start = chronotimer.now();
  func(args...);
  auto end = chronotimer.now();
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
  auto chronotimer = std::chrono::high_resolution_clock();
  auto duration = std::chrono::duration<DurationContainerType, DurationType>();
  auto start = chronotimer.now();
  for (size_t i = 0; i < iters; i++) {
    func(args...);
  }
  auto end = chronotimer.now();
  duration = (end - start) / iters;
  return duration.count();
}
// asking for parallel implementation btw

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

// as "Manhattan" norm of error-vector
template <typename T>
T accuracy(T* test, T* ref, size_t size) {
  if (test == nullptr || ref == nullptr) {
    throw std::invalid_argument("Bad pointer");
  }
  T differ;
  T res = T(0);
  for (size_t i = 0; i < size; i++) {
    differ = std::abs(test[i] - ref[i]);
    res = res + differ;
  }
  return res;
}

// as Euclidean norm of error-vector
// assume that (T)*(T)>=T(0); (T)*(T)=T(0) <=> multiplying 0s
template <typename T>
T accuracy_norm(T* test, T* ref, size_t size) {
  if (test == nullptr || ref == nullptr) {
    throw std::invalid_argument("Bad pointer");
  }
  T differ;
  T res = T(0);
  for (size_t i = 0; i < size; i++) {
    differ = std::pow(test[i] - ref[i], 2);
    res = res + differ;
  }
  // typename T should have friend sqrt() function
  return std::sqrt(res);
}

template <typename ThroughputContainer, typename DurationContainer>
class Throughput {
public:
  Throughput() {
    time = DurationContainer(0);
    throughput = ThroughputContainer(0);
  }

  template <class Function, typename... Args>
  ThroughputContainer get_tp(size_t items, Function&& f, Args&&... a) {
    time = elapsed_time<DurationContainer, std::ratio<1, 1> >(f, a...);
    throughput = ThroughputContainer(static_cast<double>(items) / time);
    return throughput;
  }

  template <class Function, typename... Args>
  ThroughputContainer get_tp_omp(size_t items, Function&& f, Args&&... a) {
    time = DurationContainer(elapsed_time_omp(f, a...));
    throughput = ThroughputContainer(static_cast<double>(items) / time);
    return throughput;
  }

  template <class Function, typename... Args>
  ThroughputContainer get_tp_avg(size_t items, size_t iterations,
                                 Function&& f, Args&&... a) {
    time = elapsed_time_avg<DurationContainer, std::ratio<1, 1> >(iterations, f, a...);
    throughput = ThroughputContainer(static_cast<double>(items) / time);
    return throughput;
  }

  template <class Function, typename... Args>
  ThroughputContainer get_tp_omp_avg(size_t items, size_t iterations,
                                     Function&& f, Args&&... a) {
    time = DurationContainer(elapsed_time_omp_avg(iterations, f, a...));
    throughput = ThroughputContainer(static_cast<double>(items) / time);
    return throughput;
  }

 private:
  DurationContainer time;
  ThroughputContainer throughput;
};
