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
};

}  // namespace it_lab_ai
