#include <string>

#include "parallel/parallel.hpp"

std::string sycl_device_name() {
  return it_lab_ai::parallel::sycl_device_name();
}
