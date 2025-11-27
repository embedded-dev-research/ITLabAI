#pragma once
#include <cstdint>

enum class Backend : uint8_t { kNaive, kOneDnn };

struct RuntimeOptions {
  Backend backend{Backend::kNaive};
  int threads{0};
  bool parallel{false};
};
