#pragma once
#include <dnnl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "layers/BinaryOpLayer.hpp"
#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class BinaryOpLayerOneDnn : public Layer {
 public:
  BinaryOpLayerOneDnn()
      : Layer(kBinaryOp), op_(BinaryOpLayer::Operation::kMul) {}
  explicit BinaryOpLayerOneDnn(BinaryOpLayer::Operation op)
      : Layer(kBinaryOp), op_(op) {}

  BinaryOpLayerOneDnn(const BinaryOpLayerOneDnn& c) : Layer(kBinaryOp) {
    this->op_ = c.op_;
  }

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

  void set_operation(BinaryOpLayer::Operation op) {
    op_ = op;
    initialized_ = false;
  }

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    Tensor a = make_tensor(v);
    return a;
  }
#endif

 private:
  BinaryOpLayer::Operation op_;
  bool initialized_ = false;
  Shape last_shape_a_;
  Shape last_shape_b_;
  Type last_type_;

  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
  std::unique_ptr<dnnl::binary> binary_prim_;
  dnnl::memory::desc src0_md_;
  dnnl::memory::desc src1_md_;
  dnnl::memory::desc dst_md_;
  Shape output_shape_;

  void initialize_onednn(const Tensor& A, const Tensor& B);
  static void validate_input(const std::vector<Tensor>& input);
  [[nodiscard]] static dnnl::memory::data_type get_dnnl_data_type(Type type);
  [[nodiscard]] static dnnl::algorithm get_dnnl_algorithm(
      BinaryOpLayer::Operation op);
  [[nodiscard]] static Shape calculate_output_shape(const Shape& shape_a,
                                                    const Shape& shape_b);
  [[nodiscard]] static bool can_broadcast(const Shape& shape_a,
                                          const Shape& shape_b);
  [[nodiscard]] static dnnl::memory::format_tag pick_format(size_t ndims);
  [[nodiscard]] static std::vector<dnnl::memory::dim> shape_to_dims(
      const Shape& shape);
};

}  // namespace it_lab_ai
