#pragma once
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace itlab_2023 {

class ReduceSumLayer : public Layer {
 public:
  explicit ReduceSumLayer(int64_t keepdims = 0);

  void run(const Tensor& input, Tensor& output) override;
  void run(const Tensor& input, const Tensor& axes, Tensor& output);

  static std::string get_name() { return "ReduceSumLayer"; }

 private:
  int64_t keepdims_;

  void normalize_axes(const Shape& input_shape,
                      std::vector<int64_t>& axes) const;
  Shape calculate_output_shape(const Shape& input_shape,
                               const std::vector<int64_t>& axes) const;

  template <typename T>
  void compute(const Tensor& input, const Shape& output_shape,
               const std::vector<int64_t>& axes, Tensor& output) const;
};

}  // namespace itlab_2023