#pragma once
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class TransposeLayer : public Layer {
 public:
  explicit TransposeLayer(std::vector<int64_t> perm = {})
      : Layer(kTranspose), perm_(std::move(perm)) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

 private:
  std::vector<int64_t> perm_;

  static void validate_perm(const Shape& input_shape,
                            const std::vector<int64_t>& perm);

  template <typename T>
  void transpose_impl(const Tensor& input, Tensor& output,
                      const std::vector<int64_t>& perm) const;
};

}  // namespace it_lab_ai