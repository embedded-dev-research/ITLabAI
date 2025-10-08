#pragma once
#include <string>

#include "layers/Layer.hpp"

namespace it_lab_ai {

std::vector<size_t> reorder(std::vector<size_t> order_vec,
                            std::vector<size_t> order);

class FlattenLayer : public Layer {
 private:
  std::vector<size_t> order_;

 public:
  FlattenLayer() : Layer(kFlatten), order_({0, 1, 2, 3}) {}
  FlattenLayer(const std::vector<size_t>& order)
      : Layer(kFlatten), order_(order) {}
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override { return Tensor(); }
#endif
};

template <typename ValueType>
void Flatten4D(const Tensor& input, Tensor& output,
               const std::vector<size_t>& order_) {
  Tensor tmp_tensor = Tensor(
      Shape({input.get_shape()[order_[0]], input.get_shape()[order_[1]],
             input.get_shape()[order_[2]], input.get_shape()[order_[3]]}),
      GetTypeEnum<ValueType>());
  std::vector<size_t> reorder_ind_vec =
      reorder(std::vector<size_t>({0, 1, 2, 3}), order_);
  std::vector<size_t> reorder_vec;
  std::vector<size_t> order_vec(4);
  for (order_vec[0] = 0; order_vec[0] < input.get_shape()[order_[0]];
       order_vec[0]++) {
    for (order_vec[1] = 0; order_vec[1] < input.get_shape()[order_[1]];
         order_vec[1]++) {
      for (order_vec[2] = 0; order_vec[2] < input.get_shape()[order_[2]];
           order_vec[2]++) {
        for (order_vec[3] = 0; order_vec[3] < input.get_shape()[order_[3]];
             order_vec[3]++) {
          reorder_vec = {
              order_vec[reorder_ind_vec[0]], order_vec[reorder_ind_vec[1]],
              order_vec[reorder_ind_vec[2]], order_vec[reorder_ind_vec[3]]};
          tmp_tensor.set<ValueType>(order_vec,
                                    input.get<ValueType>(reorder_vec));
        }
      }
    }
  }
  output = make_tensor(*tmp_tensor.as<ValueType>(),
                       Shape({input.get_shape().count()}));
}

}  // namespace it_lab_ai
