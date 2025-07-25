#include "layers/BinaryOpLayer.hpp"

namespace itlab_2023 {

namespace {
template <typename T>
T apply_binary_op(T a, T b, BinaryOpLayer::Operation op) {
  switch (op) {
    case BinaryOpLayer::Operation::MUL:
      return a * b;
    case BinaryOpLayer::Operation::ADD:
      return a + b;
    case BinaryOpLayer::Operation::SUB:
      return a - b;
    default:
      throw std::runtime_error("Unsupported binary operation");
  }
}
}  // namespace

void BinaryOpLayer::run(const Tensor& input, Tensor& output) {
  (void)input;
  (void)output;
  throw std::runtime_error(
      "Use run(const Tensor& A, const Tensor& B, Tensor& output) for binary "
      "operations");
}

void BinaryOpLayer::run(const Tensor& A, const Tensor& B, Tensor& output) {
  if (A.get_type() != B.get_type()) {
    throw std::runtime_error(
        "BinaryOpLayer: Input tensors must have the same type");
  }

  if (B.get_shape().dims() == 0 ||
      (B.get_shape().dims() == 1 && B.get_shape()[0] == 1)) {
    switch (B.get_type()) {
      case Type::kFloat:
        return run_with_scalar(A, B.as<float>()->at(0), output);
      case Type::kInt:
        return run_with_scalar(A, static_cast<float>(B.as<int>()->at(0)),
                               output);
      default:
        throw std::runtime_error("BinaryOpLayer: Unsupported scalar type");
    }
  }
  if (A.get_shape().dims() == 0 ||
      (A.get_shape().dims() == 1 && A.get_shape()[0] == 1)) {
    switch (A.get_type()) {
      case Type::kFloat:
        return run_with_scalar(B, A.as<float>()->at(0), output);
      case Type::kInt:
        return run_with_scalar(B, static_cast<float>(A.as<int>()->at(0)),
                               output);
      default:
        throw std::runtime_error("BinaryOpLayer: Unsupported scalar type");
    }
  }

  if (!can_broadcast(A.get_shape(), B.get_shape())) {
    throw std::runtime_error(
        "BinaryOpLayer: Incompatible shapes for broadcasting");
  }

  Shape output_shape =
      calculate_broadcasted_shape(A.get_shape(), B.get_shape());

  switch (A.get_type()) {
    case Type::kFloat: {
      const auto& a_data = *A.as<float>();
      const auto& b_data = *B.as<float>();
      std::vector<float> result(output_shape.count());

      for (size_t i = 0; i < result.size(); ++i) {
        size_t a_idx = get_broadcasted_index(i, A.get_shape(), output_shape);
        size_t b_idx = get_broadcasted_index(i, B.get_shape(), output_shape);
        result[i] = apply_binary_op(a_data[a_idx], b_data[b_idx], op_);
      }
      output = make_tensor(result, output_shape);
      break;
    }
    case Type::kInt: {
      const auto& a_data = *A.as<int>();
      const auto& b_data = *B.as<int>();
      std::vector<int> result(output_shape.count());

      for (size_t i = 0; i < result.size(); ++i) {
        size_t a_idx = get_broadcasted_index(i, A.get_shape(), output_shape);
        size_t b_idx = get_broadcasted_index(i, B.get_shape(), output_shape);
        result[i] = apply_binary_op(a_data[a_idx], b_data[b_idx], op_);
      }
      output = make_tensor(result, output_shape);
      break;
    }
    default:
      throw std::runtime_error("BinaryOpLayer: Unsupported tensor type");
  }
}

void BinaryOpLayer::run_with_scalar(const Tensor& input, float scalar,
                                    Tensor& output) {
  switch (input.get_type()) {
    case Type::kFloat: {
      run_with_scalar_impl<float>(input, scalar, output);
      break;
    }
    case Type::kInt: {
      run_with_scalar_impl<int>(input, static_cast<int>(scalar), output);
      break;
    }
    default:
      throw std::runtime_error(
          "BinaryOpLayer: Unsupported tensor type for scalar operation");
  }
}

template <typename ValueType>
void BinaryOpLayer::run_with_scalar_impl(const Tensor& input, ValueType scalar,
                                         Tensor& output) const {
  const auto& input_data = *input.as<ValueType>();
  std::vector<ValueType> result;
  result.reserve(input_data.size());

  for (const auto& val : input_data) {
    result.push_back(apply_binary_op(val, scalar, op_));
  }

  output = make_tensor(result, input.get_shape());
}

template <typename ValueType>
BinaryOpLayer::BinaryOpLayerImpl<ValueType>::BinaryOpLayerImpl(
    BinaryOpLayer::Operation op)
    : LayerImpl<ValueType>(Shape({1}), Shape({1})), op_(op) {}

template <typename ValueType>
std::vector<ValueType> BinaryOpLayer::BinaryOpLayerImpl<ValueType>::run(
    const std::vector<ValueType>& inputA,
    const std::vector<ValueType>& inputB) const {
  if (inputA.size() != inputB.size()) {
    throw std::runtime_error("BinaryOpLayer: Input sizes must match");
  }

  std::vector<ValueType> result(inputA.size());
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = apply_binary_op(inputA[i], inputB[i], op_);
  }
  return result;
}

bool BinaryOpLayer::can_broadcast(const Shape& shape_A, const Shape& shape_B) {
  size_t a_dims = shape_A.dims();
  size_t b_dims = shape_B.dims();
  size_t max_dims = std::max(a_dims, b_dims);

  for (size_t i = 0; i < max_dims; ++i) {
    size_t a_dim = (i < a_dims) ? shape_A[a_dims - 1 - i] : 1;
    size_t b_dim = (i < b_dims) ? shape_B[b_dims - 1 - i] : 1;

    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      return false;
    }
  }
  return true;
}

Shape BinaryOpLayer::calculate_broadcasted_shape(const Shape& shape_A,
                                                 const Shape& shape_B) {
  size_t a_dims = shape_A.dims();
  size_t b_dims = shape_B.dims();
  size_t max_dims = std::max(a_dims, b_dims);
  Shape result(max_dims);

  for (size_t i = 0; i < max_dims; ++i) {
    size_t a_dim = (i < a_dims) ? shape_A[a_dims - 1 - i] : 1;
    size_t b_dim = (i < b_dims) ? shape_B[b_dims - 1 - i] : 1;
    result[max_dims - 1 - i] = std::max(a_dim, b_dim);
  }
  return result;
}

std::vector<size_t> BinaryOpLayer::get_strides(const Shape& shape) {
  std::vector<size_t> strides(shape.dims());
  if (strides.empty()) return strides;

  strides.back() = 1;
  for (int i = (int)shape.dims() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

size_t BinaryOpLayer::get_broadcasted_index(size_t flat_index,
                                            const Shape& input_shape,
                                            const Shape& output_shape) {
  size_t input_dims = input_shape.dims();
  size_t output_dims = output_shape.dims();
  size_t index = 0;
  auto strides = get_strides(input_shape);

  for (size_t i = 0; i < output_dims; ++i) {
    size_t output_dim = output_shape[i];
    size_t input_dim = (i >= output_dims - input_dims)
                           ? input_shape[i - (output_dims - input_dims)]
                           : 1;

    if (input_dim == 1) continue;

    size_t pos_in_dim =
        (flat_index / get_strides(output_shape)[i]) % output_dim;
    if (i >= output_dims - input_dims) {
      size_t input_pos = i - (output_dims - input_dims);
      index += pos_in_dim * strides[input_pos];
    }
  }
  return index;
}

template class BinaryOpLayer::BinaryOpLayerImpl<int>;
template class BinaryOpLayer::BinaryOpLayerImpl<float>;

}  // namespace itlab_2023