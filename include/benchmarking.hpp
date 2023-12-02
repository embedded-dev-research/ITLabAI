// Using chrono for good measurements and parallelism support

#pragma once
#include <chrono>
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

// as "Manhattan" norm of error-vector
template <typename T>
T accuracy(T* test, T* ref, size_t size) {
  if (test == nullptr || ref == nullptr) {
    throw std::invalid_argument("Bad pointer");
  }
  T differ;
  T res = T(0);
  for (size_t i = 0; i < size; i++) {
    differ = test[i] - ref[i];
    if (differ < T(0)) {
      differ = -differ;
    }
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
    differ = (test[i] - ref[i]) * (test[i] - ref[i]);
    res = res + differ;
  }
  // typename T should have friend sqrt() function
  return sqrt(res);
}

