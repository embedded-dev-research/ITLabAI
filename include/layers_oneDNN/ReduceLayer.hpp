#pragma once
#include <cstdint>
#include <dnnl.hpp>
#include <memory>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/ReduceLayer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ReduceLayerOneDnn : public Layer {
 public:
  ReduceLayerOneDnn(ReduceLayer::Operation op, int64_t keepdims,
                    const std::vector<int64_t>& axes)
      : Layer(kReduce), op_(op), keepdims_(keepdims), axes_(axes) {}

  explicit ReduceLayerOneDnn(int64_t keepdims = 0,
                             const std::vector<int64_t>& axes = {})
      : ReduceLayerOneDnn(ReduceLayer::Operation::kSum, keepdims, axes) {}

  ReduceLayerOneDnn(const ReduceLayerOneDnn& c) : Layer(kReduce) {
    this->op_ = c.op_;
    this->keepdims_ = c.keepdims_;
    this->axes_ = c.axes_;
  }

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

  void set_axes(const std::vector<int64_t>& axes) {
    axes_ = axes;
    initialized_ = false;
  }

  void set_keepdims(int64_t keepdims) {
    keepdims_ = keepdims;
    initialized_ = false;
  }

  void set_operation(ReduceLayer::Operation op) {
    op_ = op;
    initialized_ = false;
  }

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    return Tensor();
  }
#endif

 private:
  ReduceLayer::Operation op_;
  int64_t keepdims_;
  std::vector<int64_t> axes_;
  std::vector<int64_t> normalized_axes_;
  std::vector<int64_t> last_axes_;

  bool initialized_ = false;
  Shape last_input_shape_;
  Type last_type_;

  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
  std::unique_ptr<dnnl::reduction> reduction_prim_;

  dnnl::memory::desc src_md_;
  dnnl::memory::desc dst_md_;
  Shape output_shape_;

  void initialize_onednn(const Tensor& input);
  static void validate_input(const std::vector<Tensor>& input);
  [[nodiscard]] static dnnl::memory::data_type get_dnnl_data_type(Type type);
  [[nodiscard]] static dnnl::algorithm get_dnnl_algorithm(
      ReduceLayer::Operation op);
  [[nodiscard]] static dnnl::memory::format_tag pick_format(size_t ndims);
  static void normalize_axes(const Shape& input_shape,
                             std::vector<int64_t>& axes);
  [[nodiscard]] Shape calculate_output_shape(
      const Shape& input_shape, const std::vector<int64_t>& axes) const;

  [[nodiscard]] static std::vector<dnnl::memory::dim> shape_to_dims(
      const Shape& shape);
  template <typename T>
  std::vector<T> remove_unit_dims(const std::vector<T>& src_data,
                                  const Shape& src_shape,
                                  const Shape& dst_shape);
};

}  // namespace it_lab_ai
