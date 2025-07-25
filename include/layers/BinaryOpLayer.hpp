#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Tensor.hpp"
#include "layers/Layer.hpp"

namespace itlab_2023 {

class BinaryOpLayer : public Layer {
 public:
  enum class Operation { MUL, ADD, SUB };

  BinaryOpLayer() = default;
  explicit BinaryOpLayer(Operation op) : op_(op) {}

  static std::string get_name() { return "Binary Operation Layer"; }
  void run(const Tensor& input, Tensor& output) override;
  void run(const Tensor& A, const Tensor& B, Tensor& output);

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    return make_tensor(v);
  }
#endif

 private:
  Operation op_ = Operation::MUL;
  std::shared_ptr<void> impl_;

  template <typename ValueType>
  void run_with_scalar_impl(const Tensor& input, ValueType scalar,
                            Tensor& output) const;
  void run_with_scalar(const Tensor& input, float scalar, Tensor& output);

  static bool can_broadcast(const Shape& shape_A, const Shape& shape_B);
  static Shape calculate_broadcasted_shape(const Shape& shape_A,
                                           const Shape& shape_B);
  static std::vector<size_t> get_strides(const Shape& shape);
  static size_t get_broadcasted_index(size_t flat_index,
                                      const Shape& input_shape,
                                      const Shape& output_shape);

  template <typename ValueType>
  class BinaryOpLayerImpl;
};

template <typename ValueType>
class BinaryOpLayer::BinaryOpLayerImpl : public LayerImpl<ValueType> {
 public:
  BinaryOpLayerImpl() = delete;
  explicit BinaryOpLayerImpl(BinaryOpLayer::Operation op);
  BinaryOpLayerImpl(const BinaryOpLayerImpl&) = default;
  BinaryOpLayerImpl& operator=(const BinaryOpLayerImpl&) = default;

  std::vector<ValueType> run(const std::vector<ValueType>& inputA,
                             const std::vector<ValueType>& inputB) const;

 private:
  BinaryOpLayer::Operation op_;
};

}  // namespace itlab_2023