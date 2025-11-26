#pragma once

enum class Backend { kNaive, kOneDnn };

struct RuntimeOptions {
  Backend backend{Backend::kNaive};
  int threads{0};
  bool parallel{false};
};
