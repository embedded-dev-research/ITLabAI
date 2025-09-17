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
  std::cout << "FlattenLayer started" << std::endl;

  if (input.size() != 1) {
    throw std::runtime_error("FlattenLayer: Input tensors not 1");
  }

  const auto& input_tensor = input[0];
  const auto& input_shape = input_tensor.get_shape();

  std::cout << "Input shape: ";
  for (size_t i = 0; i < input_shape.dims(); ++i) {
    std::cout << input_shape[i] << " ";
  }
  std::cout << ", axis: " << axis_ << std::endl;

  // Ѕезопасна€ проверка axis
  size_t axis = static_cast<size_t>(axis_);
  if (axis_ < 0) {
    axis = 0;  // защита от отрицательного axis
  }
  if (axis >= input_shape.dims()) {
    axis = input_shape.dims() - 1;  // защита от выхода за границы
  }

  // –асчет выходной формы
  size_t first_dim = 1;
  for (size_t i = 0; i < axis; ++i) {
    first_dim *= input_shape[i];
  }

  size_t second_dim = 1;
  for (size_t i = axis; i < input_shape.dims(); ++i) {
    second_dim *= input_shape[i];
  }

  Shape output_shape({first_dim, second_dim});

  // ѕростое копирование данных (flatten не мен€ет данные)
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

  std::cout << "Output shape: " << first_dim << " " << second_dim << std::endl;
  std::cout << "FlattenLayer completed" << std::endl;
}

}  // namespace it_lab_ai
