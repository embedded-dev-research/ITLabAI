#include "layers/BinaryOpLayer.hpp"

namespace it_lab_ai {

namespace {
template <typename T>
T apply_binary_op(T a, T b, BinaryOpLayer::Operation op) {
  switch (op) {
    case BinaryOpLayer::Operation::kMul:
      return a * b;
    case BinaryOpLayer::Operation::kAdd:
      return a + b;
    case BinaryOpLayer::Operation::kSub:
      return a - b;
    case BinaryOpLayer::Operation::kDiv:
      if (b == 0) throw std::runtime_error("Division by zero");
      return a / b;
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

  if (is_scalar_tensor(B)) {
    switch (B.get_type()) {
      case Type::kFloat:
        run_with_scalar(A, B.as<float>()->at(0), output);
        return;
      case Type::kInt:
        run_with_scalar(A, static_cast<float>(B.as<int>()->at(0)), output);
        return;
      default:
        throw std::runtime_error("Unsupported scalar type");
    }
  }

  if (is_scalar_tensor(A)) {
    switch (A.get_type()) {
      case Type::kFloat:
        run_with_scalar(B, A.as<float>()->at(0), output);
        return;
      case Type::kInt:
        run_with_scalar(B, static_cast<float>(A.as<int>()->at(0)), output);
        return;
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
    case Type::kFloat:
      run_broadcast_impl<float>(A, B, output, output_shape);
      break;
    case Type::kInt:
      run_broadcast_impl<int>(A, B, output, output_shape);
      break;
    default:
      throw std::runtime_error("Unsupported tensor type");
  }
}

void BinaryOpLayer::run_with_scalar(const Tensor& input, float scalar,
                                    Tensor& output) const {
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
void BinaryOpLayer::run_broadcast_impl(const Tensor& A, const Tensor& B,
                                       Tensor& output,
                                       const Shape& output_shape) const {
  const auto& a_data = *A.as<ValueType>();
  const auto& b_data = *B.as<ValueType>();
  std::vector<ValueType> result(output_shape.count());
  const auto strides_a = get_strides(A.get_shape());
  const auto strides_b = get_strides(B.get_shape());
  const auto strides_output = get_strides(output_shape);

  for (size_t i = 0; i < result.size(); ++i) {
    size_t a_idx = get_broadcasted_index(i, A.get_shape(), output_shape,
                                         strides_a, strides_output);
    size_t b_idx = get_broadcasted_index(i, B.get_shape(), output_shape,
                                         strides_b, strides_output);
    result[i] = apply_binary_op(a_data[a_idx], b_data[b_idx], op_);
  }

  output = make_tensor(result, output_shape);
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

size_t BinaryOpLayer::get_broadcasted_index(
    size_t flat_index, const Shape& input_shape, const Shape& output_shape,
    const std::vector<size_t>& input_strides,
    const std::vector<size_t>& output_strides) {
  size_t input_dims = input_shape.dims();
  size_t output_dims = output_shape.dims();
  size_t index = 0;

  for (size_t i = 0; i < output_dims; ++i) {
    size_t output_dim = output_shape[i];
    size_t input_dim = (i >= output_dims - input_dims)
                           ? input_shape[i - (output_dims - input_dims)]
                           : 1;

    if (input_dim == 1) continue;

    size_t pos_in_dim = (flat_index / output_strides[i]) % output_dim;
    if (i >= output_dims - input_dims) {
      size_t input_pos = i - (output_dims - input_dims);
      index += pos_in_dim * input_strides[input_pos];
    }
  }
  return index;
}

bool BinaryOpLayer::is_scalar_tensor(const Tensor& t) {
  const auto& shape = t.get_shape();
  const size_t dims = shape.dims();

  if (dims == 0) return true;

  for (size_t i = 0; i < dims; ++i) {
    if (shape[i] != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace it_lab_ai