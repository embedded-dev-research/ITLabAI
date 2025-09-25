#include "layers/FlattenLayer.hpp"

namespace it_lab_ai {

// reorder coords
std::vector<size_t> reorder(std::vector<size_t> order_vec,
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

//void FlattenLayer::run(const std::vector<Tensor>& input,
//                       std::vector<Tensor>& output) {
//  switch (input[0].get_type()) {
//    case Type::kInt: {
//      if (input[0].get_shape().dims() == 4) {
//        Flatten4D<int>(input[0], output[0], order_);
//      } else {
//        output[0] = make_tensor(*input[0].as<int>(),
//                                Shape({input[0].get_shape().count()}));
//      }
//      break;
//    }
//    case Type::kFloat: {
//      if (input[0].get_shape().dims() == 4) {
//        Flatten4D<float>(input[0], output[0], order_);
//      } else {
//        output[0] = make_tensor(*input[0].as<float>(),
//                                Shape({input[0].get_shape().count()}));
//      }
//      break;
//    }
//    default: {
//      throw std::runtime_error("No such type");
//    }
//  }
//}

void FlattenLayer::run(const std::vector<Tensor>& input,
                       std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("FlattenLayer: Input tensors not 1");
  }

  const auto& input_tensor = input[0];
  const auto& input_shape = input_tensor.get_shape();

  std::cout << "FlattenLayer input shape: ";
  for (size_t i = 0; i < input_shape.dims(); ++i) {
    std::cout << input_shape[i] << " ";
  }
  std::cout << std::endl;

  Shape output_shape;

  // Если задан order_ (старый стиль)
  if (!order_.empty() && order_.size() == 4) {
    // Используем существующую логику с перестановкой
    switch (input_tensor.get_type()) {
      case Type::kFloat:
        Flatten4D<float>(input_tensor, output[0], order_);
        break;
      case Type::kInt:
        Flatten4D<int>(input_tensor, output[0], order_);
        break;
      default:
        throw std::runtime_error("Unsupported tensor type");
    }
  } else {
    // Новый стиль: простой flatten в 1D
    size_t total_size = input_shape.count();
    output_shape = Shape({total_size});

    std::cout << "Simple flatten to 1D: " << total_size << std::endl;

    switch (input_tensor.get_type()) {
      case Type::kInt:
        output[0] = make_tensor(*input_tensor.as<int>(), output_shape);
        break;
      case Type::kFloat:
        output[0] = make_tensor(*input_tensor.as<float>(), output_shape);
        break;
      default:
        throw std::runtime_error("Unsupported tensor type");
    }
  }

  std::cout << "FlattenLayer output shape: ";
  for (size_t i = 0; i < output[0].get_shape().dims(); ++i) {
    std::cout << output[0].get_shape()[i] << " ";
  }
  std::cout << std::endl;
}
}  // namespace it_lab_ai
