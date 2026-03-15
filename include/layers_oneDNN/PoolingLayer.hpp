#pragma once

#include <dnnl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class PoolingLayerOneDnn : public Layer {
 public:
  explicit PoolingLayerOneDnn(const Shape& pooling_shape,
                              const Shape& strides = {2, 2},
                              const Shape& pads = {0, 0, 0, 0},
                              const Shape& dilations = {1, 1},
                              bool ceil_mode = false,
                              std::string pooling_type = "average")
      : Layer(kPooling),
        poolingShape_(pooling_shape),
        strides_(strides),
        pads_(pads),
        dilations_(dilations),
        ceil_mode_(ceil_mode),
        poolingType_(std::move(pooling_type)),
        engine_(std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0)),
        stream_(std::make_unique<dnnl::stream>(*engine_)) {}

  PoolingLayerOneDnn(const PoolingLayerOneDnn& c)
      : Layer(kPooling),
        engine_(std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0)),
        stream_(std::make_unique<dnnl::stream>(*engine_)) {
    this->poolingShape_ = c.poolingShape_;
    this->strides_ = c.strides_;
    this->pads_ = c.pads_;
    this->dilations_ = c.dilations_;
    this->ceil_mode_ = c.ceil_mode_;
    this->poolingType_ = c.poolingType_;
  }

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

  void setStrides(size_t h, size_t w) {
    strides_ = {h, w};
    initialized_ = false;
  }

  void setPads(size_t top, size_t bottom, size_t left, size_t right) {
    pads_ = {top, bottom, left, right};
    initialized_ = false;
  }

  void setDilations(size_t h, size_t w) {
    dilations_ = {h, w};
    initialized_ = false;
  }

  void setCeilMode(bool ceil_mode) {
    ceil_mode_ = ceil_mode;
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
  void initialize_onednn(const Shape& shape, Type data_type);
  [[nodiscard]] dnnl::algorithm get_PoolType() const;
  static void validate_input(const std::vector<Tensor>& input);
  [[nodiscard]] static dnnl::memory::data_type get_dnnl_data_type(Type type);
  [[nodiscard]] Shape calculate_output_shape(const Shape& input_shape) const;

  Shape poolingShape_;
  Shape strides_;
  Shape pads_;
  Shape dilations_;
  bool ceil_mode_;
  std::string poolingType_;

  bool initialized_ = false;
  Shape last_shape_;
  Type last_type_;

  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
  std::unique_ptr<dnnl::pooling_forward> pool_prim_;
  dnnl::memory::desc src_memory_desc_;
  dnnl::memory::desc dst_memory_desc_;
  Shape output_shape_;
};

}  // namespace it_lab_ai
