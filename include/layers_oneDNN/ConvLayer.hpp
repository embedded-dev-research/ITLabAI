#pragma once

#include <dnnl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class ConvLayerOneDnn : public Layer {
 public:
  ConvLayerOneDnn()
      : Layer(kConvolution),
        stride_(1),
        pads_(0),
        dilations_(1),
        kernel_(nullptr),
        bias_(nullptr),
        group_(1),
        use_legacy_(false) {}

  ConvLayerOneDnn(size_t stride, size_t pads, size_t dilations,
                  const Tensor& kernel, const Tensor& bias = Tensor(),
                  size_t group = 1, bool use_legacy = false)
      : Layer(kConvolution),
        stride_(stride),
        pads_(pads),
        dilations_(dilations),
        kernel_(std::make_shared<Tensor>(kernel)),
        bias_(std::make_shared<Tensor>(bias)),
        group_(group),
        use_legacy_(use_legacy) {}

  ConvLayerOneDnn(size_t stride, size_t pads, size_t dilations,
                  std::shared_ptr<Tensor> kernel,
                  std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(),
                  size_t group = 1, bool use_legacy = false)
      : Layer(kConvolution),
        stride_(stride),
        pads_(pads),
        dilations_(dilations),
        kernel_(std::move(kernel)),
        bias_(std::move(bias)),
        group_(group),
        use_legacy_(use_legacy) {}

  ConvLayerOneDnn(const ConvLayerOneDnn& c) : Layer(kConvolution) {
    this->stride_ = c.stride_;
    this->pads_ = c.pads_;
    this->dilations_ = c.dilations_;
    this->kernel_ = c.kernel_;
    this->bias_ = c.bias_;
    this->group_ = c.group_;
    this->use_legacy_ = c.use_legacy_;
  }

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    return *kernel_;
  }
#endif

 private:
  void initialize_convolution(const Shape& input_shape, Type data_type);
  void validate_input(const std::vector<Tensor>& input) const;
  void validate_depthwise_input(const std::vector<Tensor>& input) const;
  static void create_output_tensor(Tensor& output_tensor,
                                   const Shape& output_shape, Type data_type,
                                   dnnl::memory& dst_memory);
  static void fill_memory_with_tensor(dnnl::memory& memory,
                                      const Tensor& tensor, Type data_type);
  void initialize_special_conv(const Shape& input_shape, Type data_type);

  void run_special_conv(const std::vector<Tensor>& input,
                        std::vector<Tensor>& output);

  [[nodiscard]] static dnnl::memory::dims shape_to_dims(const Shape& shape) {
    dnnl::memory::dims dims;
    for (size_t i = 0; i < shape.dims(); ++i) {
      dims.push_back(static_cast<dnnl::memory::dim>(shape[i]));
    }
    return dims;
  }

  [[nodiscard]] static Shape dims_to_shape(const dnnl::memory::dims& dims) {
    std::vector<size_t> shape_vec;
    for (auto dim : dims) {
      shape_vec.push_back(static_cast<size_t>(dim));
    }
    return Shape(shape_vec);
  }

  template <typename T>
  std::vector<T> reorder_hwio_to_oihw(const Tensor& kernel);

  [[nodiscard]] Shape get_output_shape(const Shape& input_shape) const;

  [[nodiscard]] dnnl::memory::dims get_output_dims(
      const Shape& input_shape) const {
    return shape_to_dims(get_output_shape(input_shape));
  }

  [[nodiscard]] dnnl::memory::dims get_kernel_dims() const;

  [[nodiscard]] bool is_depthwise_convolution() const;

  size_t stride_;
  size_t pads_;
  size_t dilations_;
  std::shared_ptr<Tensor> kernel_;
  std::shared_ptr<Tensor> bias_;
  size_t group_;
  bool use_legacy_;

  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;

  std::unique_ptr<dnnl::convolution_forward> conv_prim_;
  dnnl::memory src_memory_;
  dnnl::memory weights_memory_;
  dnnl::memory bias_memory_;
  dnnl::memory dst_memory_;

  std::unique_ptr<dnnl::convolution_forward> depthwise_conv_prim_;
  dnnl::memory depthwise_src_memory_;
  dnnl::memory depthwise_weights_memory_;
  dnnl::memory depthwise_bias_memory_;
  dnnl::memory depthwise_dst_memory_;

  bool initialized_ = false;
  Shape last_input_shape_;
  Type last_data_type_;
};

}  // namespace it_lab_ai
