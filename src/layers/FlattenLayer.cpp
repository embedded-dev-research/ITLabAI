#include "layers/FlattenLayer.hpp"

namespace itlab_2023 {

// reorder coords
std::vector<size_t> reordered(std::vector<size_t> order_vec,
                              std::vector<size_t> order) {
  size_t min_ind;
  for (size_t i = 0; i < order.size() - 1; i++) {
    min_ind = i;
    for (size_t j = i + 1; j < order.size(); j++) {
      if (order[j] < order[min_ind]) {
        min_ind = j;
      }
    }
    std::swap(order_vec[i], order_vec[min_ind]);
    std::swap(order[i], order[min_ind]);
  }
  return order_vec;
}

void FlattenLayer::run(const Tensor &input, Tensor &output) {
  switch (input.get_type()) {
    case Type::kInt: {
      if (input.get_shape().dims() == 4) {
        Tensor tmp_tensor = Tensor(
            Shape({input.get_shape()[order[0]], input.get_shape()[order[1]],
                   input.get_shape()[order[2]], input.get_shape()[order[3]]}),
            Type::kFloat);
        std::vector<size_t> reorder_vec(4);
        std::vector<size_t> order_vec(4);
        for (order_vec[0] = 0; order_vec[0] < input.get_shape()[order[0]];
             order_vec[0]++) {
          for (order_vec[1] = 0; order_vec[1] < input.get_shape()[order[1]];
               order_vec[1]++) {
            for (order_vec[2] = 0; order_vec[2] < input.get_shape()[order[2]];
                 order_vec[2]++) {
              for (order_vec[3] = 0; order_vec[3] < input.get_shape()[order[3]];
                   order_vec[3]++) {
                reorder_vec = reordered(order_vec, order);
                tmp_tensor.set<int>(order_vec, input.get<int>(reorder_vec));
              }
            }
          }
        }
        output = make_tensor(*tmp_tensor.as<int>(),
                             Shape({input.get_shape().count()}));
      } else {
        output =
            make_tensor(*input.as<int>(), Shape({input.get_shape().count()}));
      }
      break;
    }
    case Type::kFloat: {
      if (input.get_shape().dims() == 4) {
        Tensor tmp_tensor = Tensor(
            Shape({input.get_shape()[order[0]], input.get_shape()[order[1]],
                   input.get_shape()[order[2]], input.get_shape()[order[3]]}),
            Type::kFloat);
        std::vector<size_t> reorder_vec(4);
        std::vector<size_t> order_vec(4);
        for (order_vec[0] = 0; order_vec[0] < input.get_shape()[order[0]];
             order_vec[0]++) {
          for (order_vec[1] = 0; order_vec[1] < input.get_shape()[order[1]];
               order_vec[1]++) {
            for (order_vec[2] = 0; order_vec[2] < input.get_shape()[order[2]];
                 order_vec[2]++) {
              for (order_vec[3] = 0; order_vec[3] < input.get_shape()[order[3]];
                   order_vec[3]++) {
                reorder_vec = reordered(order_vec, order);
                tmp_tensor.set<float>(order_vec, input.get<float>(reorder_vec));
              }
            }
          }
        }
        output = make_tensor(*tmp_tensor.as<float>(),
                             Shape({input.get_shape().count()}));
      } else {
        output =
            make_tensor(*input.as<float>(), Shape({input.get_shape().count()}));
      }
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }
}
}  // namespace itlab_2023
