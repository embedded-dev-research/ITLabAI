#pragma once
#include <functional>
#include <numeric>

#include "Tensor.hpp"

namespace itlab_2023 {

class MulLayer {
 public:
  MulLayer() = default;

  void run(const Tensor& A, const Tensor& B, Tensor& output);

  static void run_with_scalar(const Tensor& input, float scalar,
                              Tensor& output);

  static std::string get_name() { return "Element-wise Multiplication Layer"; }

 private:
  static bool can_broadcast(const Shape& shape_A, const Shape& shape_B);
  static Shape calculate_broadcasted_shape(const Shape& shape_A,
                                           const Shape& shape_B);
  static std::vector<size_t> get_strides(const Shape& shape);
  size_t get_broadcasted_index(size_t flat_index, const Shape& input_shape,
                               const Shape& output_shape);
};

}  // namespace itlab_2023