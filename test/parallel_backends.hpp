#pragma once

#include <string>
#include <vector>

#include "graph/runtime_options.hpp"

namespace it_lab_ai {
namespace test_support {

inline std::vector<ParBackend> all_parallel_backends() {
  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};
#ifdef ITLABAI_HAS_SYCL
  backends.push_back(ParBackend::kSycl);
#endif
  return backends;
}

inline RuntimeOptions make_options(ParBackend backend) {
  RuntimeOptions options;
  options.backend = Backend::kNaive;
  options.par_backend = backend;
  return options;
}

inline std::vector<RuntimeOptions> all_parallel_options() {
  std::vector<RuntimeOptions> options;
  for (ParBackend backend : all_parallel_backends()) {
    options.push_back(make_options(backend));
  }
  return options;
}

inline std::string parallel_backend_name(ParBackend backend) {
  switch (backend) {
    case ParBackend::kTbb:
      return "TBB";
    case ParBackend::kThreads:
      return "STL";
    case ParBackend::kOmp:
      return "OMP";
    case ParBackend::kKokkos:
      return "Kokkos";
    case ParBackend::kSycl:
      return "SYCL";
    case ParBackend::kSeq:
      return "Seq";
  }
  return "Unknown";
}

}  // namespace test_support
}  // namespace it_lab_ai
