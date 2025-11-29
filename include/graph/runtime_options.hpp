#pragma once
#include <cstdint>

enum class Backend : uint8_t { kNaive, kOneDnn };
enum class ParallelBackend : uint8_t {
  kNone,
  kTBB,
  kSTL,
  kOMP,
  kKokkos,
  kSycl
};

struct RuntimeOptions {
  Backend backend{Backend::kNaive};
  ParallelBackend parallel_backend{ParallelBackend::kNone};
  int threads{0};
  bool parallel{false};
};
