#pragma once
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ReshapeLayer : public Layer {
 public:
  explicit ReshapeLayer(bool allowzero = false,
                        const std::vector<int64_t>& shape = {})
      : Layer(kReshape), allowzero_(allowzero), shape_(shape) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

  void set_shape(const std::vector<int64_t>& shape) { shape_ = shape; }
  void set_allowzero(bool allowzero) { allowzero_ = allowzero; }

 private:
  bool allowzero_;
  std::vector<int64_t> shape_;

  template <typename T>
  void reshape_impl(const Tensor& input, Tensor& output,
                    const std::vector<int64_t>& target_shape,
                    const std::vector<int64_t>& final_shape) const;
  template <typename T>
  void apply_per_batch_reshape(const Tensor& input, Tensor& output,
                               const std::vector<int64_t>& target_shape) const;
  static std::vector<int64_t> calculate_output_shape(
      const Shape& input_shape, const std::vector<int64_t>& requested_shape);
};

}  // namespace it_lab_ai