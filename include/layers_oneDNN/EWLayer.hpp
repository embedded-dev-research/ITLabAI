#pragma once

#include <dnnl.hpp>
#include <memory>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class EwLayerOneDnn : public Layer {
 public:
  EwLayerOneDnn()
      : Layer(kElementWise), func_("none"), alpha_(0.0F), beta_(0.0F) {}

  EwLayerOneDnn(std::string function, float alpha = 0.0F, float beta = 0.0F)
      : Layer(kElementWise),
        func_(std::move(function)),
        alpha_(alpha),
        beta_(beta) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
  static bool is_function_supported(const std::string& function);

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    Tensor a = make_tensor(v);
    return a;
  }
#endif

 private:
  void initialize_onednn(const Shape& shape, Type data_type);
  dnnl::algorithm get_algorithm() const;
  void validate_input(const std::vector<Tensor>& input) const;

  std::string func_;
  float alpha_;
  float beta_;

  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
  std::unique_ptr<dnnl::eltwise_forward> eltwise_prim_;
  dnnl::memory::desc memory_desc_;
  bool initialized_ = false;
};

}  // namespace it_lab_ai