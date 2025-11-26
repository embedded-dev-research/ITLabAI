#pragma once

enum class Backend { Naive, OneDnn };

struct RuntimeOptions {
  Backend backend{Backend::Naive};
  int threads{0};
  bool parallel{false};
};
