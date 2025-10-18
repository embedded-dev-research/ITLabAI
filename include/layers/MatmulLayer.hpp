#pragma once
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class MatmulLayer : public Layer {
 public:
  MatmulLayer() : Layer(kMatmul) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif

 private:
  template <typename T>
  void matmul_impl(const Tensor& a, const Tensor& b, Tensor& output) const;

  template <typename T>
  void matmul_1d_1d(const Tensor& a, const Tensor& b, Tensor& output) const;

  template <typename T>
  void matmul_1d_2d(const Tensor& a, const Tensor& b, Tensor& output) const;

  template <typename T>
  void matmul_2d_1d(const Tensor& a, const Tensor& b, Tensor& output) const;

  template <typename T>
  void matmul_2d_2d(const Tensor& a, const Tensor& b, Tensor& output) const;

  template <typename T>
  void matmul_nd_nd(const Tensor& a, const Tensor& b, Tensor& output) const;
};

}  // namespace it_lab_ai