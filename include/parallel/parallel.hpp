#pragma once
#include "backends.hpp"

namespace it_lab_ai {
namespace parallel {

constexpr bool has_omp =
#ifdef HAS_OPENMP
    true;
#else
    false;
#endif

inline Backend resolve_default_backend(std::size_t n, const Options& opt) {
  if (n < opt.min_parallel_n) {
    return Backend::Seq;
  }

#ifdef HAS_OPENMP
  return Backend::OMP;
#else
  return Backend::TBB;
#endif
}

inline Backend select_backend(const Options& opt, std::size_t n) {
  if (opt.backend != Backend::Seq && n < opt.min_parallel_n) {
    return Backend::Seq;
  }

  if (opt.backend == Backend::Seq || opt.backend == Backend::Threads ||
      opt.backend == Backend::TBB || opt.backend == Backend::OMP) {
    return opt.backend;
  }

  return resolve_default_backend(n, opt);
}

template <typename Func>
inline void parallel_for(std::size_t count, Func&& func,
                         const Options& opt = {}) {
  if (count == 0) return;

  Backend backend = select_backend(opt, count);

  switch (backend) {
    case Backend::Seq:
      impl_seq(count, std::forward<Func>(func));
      break;
    case Backend::Threads:
      impl_threads(count, std::forward<Func>(func), opt);
      break;
    case Backend::TBB:
      impl_tbb(count, std::forward<Func>(func), opt);
      break;
    case Backend::OMP:
      impl_omp(count, std::forward<Func>(func), opt);
      break;
  }
}

template <typename Func>
inline void parallel_for(std::size_t count, Func&& func, Backend backend) {
  Options opt;
  opt.backend = backend;
  parallel_for(count, std::forward<Func>(func), opt);
}

template <typename Func>
inline void parallel_for(int count, Func&& func, const Options& opt = {}) {
  if (count <= 0) return;
  parallel_for(static_cast<std::size_t>(count), std::forward<Func>(func), opt);
}

}  // namespace parallel
}  // namespace it_lab_ai