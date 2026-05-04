#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "graph/runtime_options.hpp"
#include "layers/EWLayer.hpp"
#include "layers/Tensor.hpp"

namespace {

bool almost_equal(float lhs, float rhs) {
  return std::fabs(lhs - rhs) < 1.0e-6F;
}

}  // namespace

std::string sycl_device_name();

int main() {
  try {
    using namespace it_lab_ai;

    std::vector<float> input_values = {-2.0F, -1.0F, 0.0F, 1.0F, 2.0F};
    Tensor input = make_tensor(input_values);
    std::vector<Tensor> inputs = {input};
    std::vector<Tensor> outputs(1);

    EWLayer relu("relu");
    RuntimeOptions sycl_options;
    sycl_options.par_backend = ParBackend::kSycl;
    relu.run(inputs, outputs, sycl_options);

    const std::vector<float>& relu_output = *outputs.front().as<float>();
    const std::vector<float> expected_relu = {0.0F, 0.0F, 0.0F, 1.0F, 2.0F};

    if (relu_output != expected_relu) {
      std::cerr << "ITLabAI EWLayer produced an unexpected result" << '\n';
      return 1;
    }

    const std::size_t count = relu_output.size();
    std::cout << "SYCL device: " << sycl_device_name() << '\n';

    EWLayer linear("linear", 2.0F, 1.0F);
    std::vector<Tensor> sycl_inputs = {outputs.front()};
    std::vector<Tensor> sycl_outputs(1);
    linear.run(sycl_inputs, sycl_outputs, sycl_options);

    const std::vector<float> expected_sycl = {1.0F, 1.0F, 1.0F, 3.0F, 5.0F};
    const std::vector<float>& sycl_output = *sycl_outputs.front().as<float>();
    for (std::size_t i = 0; i < count; ++i) {
      if (!almost_equal(sycl_output[i], expected_sycl[i])) {
        std::cerr << "SYCL kernel verification failed at index " << i << '\n';
        return 1;
      }
    }

    std::cout << "SYCL example completed successfully" << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }
}
