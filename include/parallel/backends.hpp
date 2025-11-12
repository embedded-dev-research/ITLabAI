#pragma once
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#ifdef HAS_OPENMP
// #include <omp.h>
#endif

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>

namespace it_lab_ai {
namespace parallel {

enum class Backend { Seq = 0, Threads = 1, TBB = 2, OMP = 3 };

struct Options {
  Backend backend = Backend::Seq;
  int max_threads = 0;
  std::size_t min_parallel_n = 1000;
  std::size_t grain = 1024;
};

inline void impl_seq(std::size_t count, std::function<void(std::size_t)> func) {
  for (std::size_t i = 0; i < count; ++i) {
    func(i);
  }
  std::cout << "Seq " << std::endl;
}

inline void impl_threads(std::size_t count,
                         std::function<void(std::size_t)> func,
                         const Options& opt) {
  int num_threads = opt.max_threads > 0
                        ? opt.max_threads
                        : static_cast<int>(std::thread::hardware_concurrency());
  if (num_threads == 0) num_threads = 4;

  std::size_t min_chunk_size = std::max(opt.grain, count / (num_threads * 4));
  if (count / num_threads < min_chunk_size) {
    num_threads = std::max(1, static_cast<int>(count / min_chunk_size));
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::size_t chunk_size = count / num_threads;
  std::size_t remainder = count % num_threads;

  std::size_t start = 0;
  for (int t = 0; t < num_threads; ++t) {
    std::size_t end =
        start + chunk_size + (t < static_cast<int>(remainder) ? 1 : 0);
    if (start >= end) break;

    threads.emplace_back([start, end, &func]() {
      for (std::size_t i = start; i < end; ++i) {
        func(i);
      }
    });

    start = end;
  }

  for (auto& thread : threads) {
    thread.join();
  }
  std::cout << "Stl " << std::endl;
}

inline void impl_tbb(std::size_t count, std::function<void(std::size_t)> func,
                     const Options& opt) {
  std::cout << "tbb " << std::endl;
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<std::size_t>(0, count, opt.grain),
      [&](const oneapi::tbb::blocked_range<std::size_t>& range) {
        for (std::size_t i = range.begin(); i < range.end(); ++i) {
          func(i);
        }
      },
      oneapi::tbb::auto_partitioner());
}

#ifdef HAS_OPENMP
inline void impl_omp(std::size_t count, std::function<void(std::size_t)> func,
                     const Options& opt) {
  if (count == 0) return;

  int num_threads = opt.max_threads > 0
                        ? opt.max_threads
                        : static_cast<int>(std::thread::hardware_concurrency());
  int chunk_size =
      static_cast<int>(std::max(opt.grain, count / (num_threads * 8)));

  int int_count = static_cast<int>(count);
  if (int_count < 0 || static_cast<std::size_t>(int_count) != count) {
    impl_seq(count, func);
    return;
  }

#pragma omp parallel for schedule(static, chunk_size) num_threads(num_threads)
  for (int i = 0; i < int_count; ++i) {
    func(static_cast<std::size_t>(i));
  }
  std::cout << "OMP " << std::endl;
}
#else
inline void impl_omp(std::size_t count, std::function<void(std::size_t)> func,
                     const Options& opt) {
  impl_seq(count, func);
}
#endif

}  // namespace parallel
}  // namespace it_lab_ai