// Using chrono for good measurements and parallelism support

#pragma once
#include <chrono>

template<typename DurationContainerType, typename DurationType, class Function,
         class... Args>
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
          class... Args>
DurationContainerType elapsed_time_avg(const size_t iters, Function&& func, Args&&... args) {
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