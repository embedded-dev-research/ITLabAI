#pragma once
#include <cstdint>

#include "parallel/parallel.hpp"

namespace it_lab_ai {

enum class Backend : uint8_t { kNaive, kOneDnn };
using ParBackend = parallel::Backend;

struct RuntimeOptions {
  Backend backend{Backend::kNaive};
  ParBackend par_backend{ParBackend::kSeq};
  int threads{0};
  bool parallel{false};

  ParBackend getEffectiveParBackend() const {
    return parallel ? par_backend : ParBackend::kSeq;
  }

  void setParallelBackend(ParBackend p) {
    par_backend = p;
    parallel = (p != ParBackend::kSeq);
  }

  bool isParallel() const {
    return parallel && (par_backend != ParBackend::kSeq);
  }
};

}  // namespace it_lab_ai
