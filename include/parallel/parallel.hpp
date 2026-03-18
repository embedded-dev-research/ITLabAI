#pragma once
#include "backends.hpp"

namespace it_lab_ai {
namespace parallel {

constexpr bool kHasOmp =
#ifdef HAS_OPENMP
    true;
#else
    false;
#endif

inline Backend resolve_default_backend(std::size_t n, const Options& opt) {
  if (n < opt.min_parallel_n) {
    return Backend::kSeq;
  }

#ifdef HAS_OPENMP
  return Backend::kOmp;
#else
  return Backend::kTbb;
#endif
}

inline Backend select_backend(const Options& opt, std::size_t n) {
  if (opt.backend != Backend::kSeq && n < opt.min_parallel_n) {
    return Backend::kSeq;
  }

  if (opt.backend == Backend::kSeq || opt.backend == Backend::kThreads ||
      opt.backend == Backend::kTbb || opt.backend == Backend::kOmp ||
      opt.backend == Backend::kKokkos) {
    return opt.backend;
  }

  return resolve_default_backend(n, opt);
}

template <typename Func>
inline void parallel_for(std::size_t count, Func&& func,
                         const Options& opt = {}) {
  if (count == 0) {
    return;
  }

  Backend backend = select_backend(opt, count);

  switch (backend) {
    case Backend::kSeq:
      impl_seq(count, std::forward<Func>(func));
      break;
    case Backend::kThreads:
      impl_threads(count, std::forward<Func>(func), opt);
      break;
    case Backend::kTbb:
      impl_tbb(count, std::forward<Func>(func), opt);
      break;
    case Backend::kOmp:
      impl_omp(count, std::forward<Func>(func), opt);
      break;
    case Backend::kKokkos:
      impl_kokkos(count, std::forward<Func>(func), opt);
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
  if (count <= 0) {
    return;
  }
  parallel_for(static_cast<std::size_t>(count), std::forward<Func>(func), opt);
}

}  // namespace parallel
}  // namespace it_lab_ai
