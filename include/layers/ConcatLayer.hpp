#pragma once
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ConcatLayer : public Layer {
 public:
  explicit ConcatLayer(int64_t axis = 0) : Layer(kConcat), axis_(axis) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  void setInputOrder(const std::vector<int>& order) { input_order_ = order; }

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

 private:
  int64_t axis_;
  std::vector<int> input_order_;
  void validate_inputs(const std::vector<Tensor>& inputs) const;
  int64_t normalize_axis(size_t rank) const;
  Shape calculate_output_shape(const std::vector<Tensor>& inputs) const;
  std::vector<Tensor> reorderInputs(const std::vector<Tensor>& inputs) const;
  template <typename T>
  void concatenate(const std::vector<Tensor>& inputs, Tensor& output) const {
    std::vector<Tensor> ordered_inputs = reorderInputs(inputs);
    Shape output_shape = calculate_output_shape(inputs);
    std::vector<T> output_data(output_shape.count(), 0);

    const int64_t axis = normalize_axis(inputs[0].get_shape().dims());
    const size_t outer_size = [&]() {
      size_t size = 1;
      for (int64_t i = 0; i < axis; ++i) {
        size *= output_shape[i];
      }
      return size;
    }();

    const size_t inner_size = [&]() {
      size_t size = 1;
      for (size_t i = axis + 1; i < output_shape.dims(); ++i) {
        size *= output_shape[i];
      }
      return size;
    }();

    size_t output_offset = 0;

    for (const auto& input : inputs) {
      const auto& input_data = *input.as<T>();
      const Shape& input_shape = input.get_shape();
      const size_t input_axis_size = input_shape[axis];

      for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t a = 0; a < input_axis_size; ++a) {
          for (size_t inner = 0; inner < inner_size; ++inner) {
            size_t input_pos =
                outer * input_axis_size * inner_size + a * inner_size + inner;

            size_t output_pos = outer * output_shape[axis] * inner_size +
                                (output_offset + a) * inner_size + inner;

            output_data[output_pos] = input_data[input_pos];
          }
        }
      }

      output_offset += input_axis_size;
    }

    output = make_tensor(output_data, output_shape);
  }
};

}  // namespace it_lab_ai