#include "layers/MulLayer.hpp"

namespace itlab_2023 {

void MulLayer::run(const Tensor& A, const Tensor& B, Tensor& output) {
  if (B.get_shape().dims() == 0 ||
      (B.get_shape().dims() == 1 && B.get_shape()[0] == 1)) {
    if (B.get_type() == Type::kFloat) {
      run_with_scalar(A, B.as<float>()->at(0), output);
    } else {
      run_with_scalar(A, static_cast<float>(B.as<int>()->at(0)), output);
    }
    return;
  }

  if (A.get_shape().dims() == 0 ||
      (A.get_shape().dims() == 1 && A.get_shape()[0] == 1)) {
    if (A.get_type() == Type::kFloat) {
      run_with_scalar(B, A.as<float>()->at(0), output);
    } else {
      run_with_scalar(B, static_cast<float>(A.as<int>()->at(0)), output);
    }
    return;
  }

  if (!can_broadcast(A.get_shape(), B.get_shape())) {
    throw std::runtime_error("MulLayer: Incompatible shapes for broadcasting");
  }

  if (A.get_shape() == B.get_shape()) {
    switch (A.get_type()) {
      case Type::kFloat: {
        const auto& a_data = *A.as<float>();
        const auto& b_data = *B.as<float>();
        std::vector<float> result;
        result.reserve(a_data.size());
        std::transform(a_data.begin(), a_data.end(), b_data.begin(),
                       std::back_inserter(result), std::multiplies<float>());
        output = make_tensor(result, A.get_shape());
        break;
      }
      case Type::kInt: {
        const auto& a_data = *A.as<int>();
        const auto& b_data = *B.as<int>();
        std::vector<int> result;
        result.reserve(a_data.size());
        std::transform(a_data.begin(), a_data.end(), b_data.begin(),
                       std::back_inserter(result), std::multiplies<int>());
        output = make_tensor(result, A.get_shape());
        break;
      }
      default:
        throw std::runtime_error("MulLayer: Unsupported tensor type");
    }
    return;
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
        result[i] = a_data[a_idx] * b_data[b_idx];
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
        result[i] = a_data[a_idx] * b_data[b_idx];
      }
      output = make_tensor(result, output_shape);
      break;
    }
    default:
      throw std::runtime_error("MulLayer: Unsupported tensor type");
  }
}

void MulLayer::run_with_scalar(const Tensor& input, float scalar,
                               Tensor& output) {
  const auto& shape = input.get_shape();
  switch (input.get_type()) {
    case Type::kFloat: {
      const auto& input_data = *input.as<float>();
      std::vector<float> result;
      result.reserve(shape.count());
      for (auto val : input_data) {
        result.push_back(val * scalar);
      }
      output = make_tensor(result, shape);
      break;
    }
    case Type::kInt: {
      const auto& input_data = *input.as<int>();
      std::vector<int> result;
      result.reserve(shape.count());
      for (auto val : input_data) {
        result.push_back(val * static_cast<int>(scalar));
      }
      output = make_tensor(result, shape);
      break;
    }
    default:
      throw std::runtime_error(
          "MulLayer: Unsupported tensor type for scalar multiplication");
  }
}

bool MulLayer::can_broadcast(const Shape& shape_A, const Shape& shape_B) {
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

Shape MulLayer::calculate_broadcasted_shape(const Shape& shape_A,
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

std::vector<size_t> MulLayer::get_strides(const Shape& shape) {
  std::vector<size_t> strides(shape.dims());
  if (strides.empty()) return strides;

  strides.back() = 1;
  for (int i = (int)shape.dims() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

size_t MulLayer::get_broadcasted_index(size_t flat_index,
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

}  // namespace itlab_2023