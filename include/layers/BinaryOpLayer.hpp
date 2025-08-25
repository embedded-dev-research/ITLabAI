#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Tensor.hpp"
#include "layers/Layer.hpp"

namespace it_lab_ai {

class BinaryOpLayer : public Layer {
 public:
  enum class Operation : uint8_t { kMul, kAdd, kSub, kDiv };

  BinaryOpLayer() = default;
  explicit BinaryOpLayer(Operation op) : op_(op) {}

  static std::string get_name() { return "Binary Operation Layer"; }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  static bool is_scalar_tensor(const Tensor& t);

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    return make_tensor(v);
  }
#endif

 private:
  Operation op_ = Operation::kMul;

  template <typename ValueType>
  void run_with_scalar_impl(const Tensor& input, ValueType scalar,
                            Tensor& output) const;
  template <typename ValueType>
  void run_broadcast_impl(const Tensor& A, const Tensor& B, Tensor& output,
                          const Shape& output_shape) const;
  void run_with_scalar(const Tensor& input, float scalar, Tensor& output) const;

  static bool can_broadcast(const Shape& shape_A, const Shape& shape_B);
  static Shape calculate_broadcasted_shape(const Shape& shape_A,
                                           const Shape& shape_B);
  static std::vector<size_t> get_strides(const Shape& shape);
  static size_t get_broadcasted_index(
      size_t flat_index, const Shape& input_shape, const Shape& output_shape,
      const std::vector<size_t>& input_strides,
      const std::vector<size_t>& output_strides);

  template <typename ValueType>
  class BinaryOpLayerImpl;
};

}  // namespace it_lab_ai