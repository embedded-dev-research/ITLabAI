#include "layers/MatmulLayer.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace it_lab_ai {

void MatmulLayer::run(const std::vector<Tensor>& input,
                      std::vector<Tensor>& output) {
  if (input.size() != 2) {
    throw std::runtime_error("MatMulLayer: Exactly 2 input tensors required");
  }
  const auto& a = input[0];
  const auto& b = input[1];

  try {
    bool should_swap = false;

    const auto& a_shape = a.get_shape();
    const auto& b_shape = b.get_shape();

    if (a_shape.dims() >= 2 && b_shape.dims() >= 2) {
      size_t a_rows = a_shape[a_shape.dims() - 2];
      size_t a_cols = a_shape[a_shape.dims() - 1];
      size_t b_rows = b_shape[b_shape.dims() - 2];
      size_t b_cols = b_shape[b_shape.dims() - 1];

      if (b_rows > a_rows) {
        should_swap = true;
      } else if (b_rows == a_rows && b_cols > a_cols) {
        should_swap = true;
      } else if (b_rows == a_rows && b_cols == a_cols) {
        size_t a_batch = 1;
        size_t b_batch = 1;
        for (size_t i = 0; i < a_shape.dims() - 2; ++i) a_batch *= a_shape[i];
        for (size_t i = 0; i < b_shape.dims() - 2; ++i) b_batch *= b_shape[i];

        if (b_batch > a_batch) {
          should_swap = true;
        }
      }
    }

    switch (a.get_type()) {
      case Type::kFloat:
        if (should_swap) {
          matmul_impl<float>(b, a, output[0]);
        } else {
          matmul_impl<float>(a, b, output[0]);
        }
        break;
      case Type::kInt:
        if (should_swap) {
          matmul_impl<int>(b, a, output[0]);
        } else {
          matmul_impl<int>(a, b, output[0]);
        }
        break;
      default:
        throw std::runtime_error("Unsupported tensor data type for MatMul");
    }
  } catch (const std::exception& e) {
    std::cerr << "ERROR in MatMul: " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "UNKNOWN ERROR in MatMul" << std::endl;
    throw;
  }
}

template <typename T>
void MatmulLayer::matmul_impl(const Tensor& a, const Tensor& b,
                              Tensor& output) const {
  const auto* a_data = a.as<T>();
  const auto* b_data = b.as<T>();

  if (!a_data || !b_data) {
    throw std::runtime_error("MatMul: Invalid input data");
  }

  const auto& a_shape = a.get_shape();
  const auto& b_shape = b.get_shape();
  size_t a_dims = a_shape.dims();
  size_t b_dims = b_shape.dims();

  if (a_dims == 1 && b_dims == 1) {
    matmul_1d_1d<T>(a, b, output);
  } else if (a_dims == 1 && b_dims >= 2) {
    matmul_1d_2d<T>(a, b, output);
  } else if (a_dims >= 2 && b_dims == 1) {
    matmul_2d_1d<T>(a, b, output);
  } else if (a_dims == 2 && b_dims == 2) {
    matmul_2d_2d<T>(a, b, output);
  } else {
    matmul_nd_nd<T>(a, b, output);
  }
}

template <typename T>
void MatmulLayer::matmul_1d_1d(const Tensor& a, const Tensor& b,
                               Tensor& output) const {
  const auto* a_data = a.as<T>();
  const auto* b_data = b.as<T>();

  if (a.get_shape()[0] != b.get_shape()[0]) {
    throw std::runtime_error("MatMul: Incompatible 1D tensor sizes");
  }

  T result = T(0);
  for (size_t i = 0; i < a.get_shape()[0]; ++i) {
    result += (*a_data)[i] * (*b_data)[i];
  }

  output = make_tensor(std::vector<T>{result}, {});
}

template <typename T>
void MatmulLayer::matmul_1d_2d(const Tensor& a, const Tensor& b,
                               Tensor& output) const {
  const auto* a_data = a.as<T>();

  const auto& b_shape = b.get_shape();
  size_t b_dims = b_shape.dims();

  if (a.get_shape()[0] != b_shape[b_dims - 2]) {
    throw std::runtime_error(
        "MatMul: Incompatible dimensions for 1D * ND multiplication");
  }

  std::vector<size_t> temp_a_shape = {1, a.get_shape()[0]};
  Tensor temp_a = make_tensor(*a_data, temp_a_shape);

  Tensor temp_output;
  matmul_nd_nd<T>(temp_a, b, temp_output);

  const auto& temp_shape = temp_output.get_shape();

  std::vector<size_t> final_shape;
  for (size_t i = 1; i < temp_shape.dims(); ++i) {
    final_shape.push_back(temp_shape[i]);
  }

  output = make_tensor(*temp_output.as<T>(), final_shape);
}

template <typename T>
void MatmulLayer::matmul_2d_1d(const Tensor& a, const Tensor& b,
                               Tensor& output) const {
  const auto* b_data = b.as<T>();

  const auto& a_shape = a.get_shape();
  size_t a_dims = a_shape.dims();

  if (a_shape[a_dims - 1] != b.get_shape()[0]) {
    throw std::runtime_error(
        "MatMul: Incompatible dimensions for ND * 1D multiplication");
  }

  std::vector<size_t> temp_b_shape = {b.get_shape()[0], 1};
  Tensor temp_b = make_tensor(*b_data, temp_b_shape);

  Tensor temp_output;
  matmul_nd_nd<T>(a, temp_b, temp_output);

  const auto& temp_shape = temp_output.get_shape();

  std::vector<size_t> final_shape;
  for (size_t i = 0; i < temp_shape.dims() - 1; ++i) {
    final_shape.push_back(temp_shape[i]);
  }

  output = make_tensor(*temp_output.as<T>(), final_shape);
}

template <typename T>
void MatmulLayer::matmul_2d_2d(const Tensor& a, const Tensor& b,
                               Tensor& output) const {
  const auto* a_data = a.as<T>();
  const auto* b_data = b.as<T>();

  const auto& a_shape = a.get_shape();
  const auto& b_shape = b.get_shape();

  if (a_shape[1] != b_shape[0]) {
    throw std::runtime_error("MatMul: Incompatible matrix dimensions");
  }

  size_t m = a_shape[0];
  size_t n = b_shape[1];
  size_t k = a_shape[1];

  std::vector<T> output_values(m * n, T(0));

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T sum = T(0);
      for (size_t l = 0; l < k; ++l) {
        sum += (*a_data)[i * k + l] * (*b_data)[l * n + j];
      }
      output_values[i * n + j] = sum;
    }
  }

  output = make_tensor(output_values, {m, n});
}

template <typename T>
void MatmulLayer::matmul_nd_nd(const Tensor& a, const Tensor& b,
                               Tensor& output) const {
  const auto* a_data = a.as<T>();
  const auto* b_data = b.as<T>();

  const auto& a_shape = a.get_shape();
  const auto& b_shape = b.get_shape();
  size_t a_dims = a_shape.dims();
  size_t b_dims = b_shape.dims();

  if (a_shape[a_dims - 1] != b_shape[b_dims - 2]) {
    throw std::runtime_error("MatMul: Incompatible matrix dimensions");
  }

  size_t batch_dims_a = (a_dims >= 2) ? a_dims - 2 : 0;
  size_t batch_dims_b = (b_dims >= 2) ? b_dims - 2 : 0;
  size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

  std::vector<size_t> batch_shape_a(max_batch_dims, 1);
  std::vector<size_t> batch_shape_b(max_batch_dims, 1);

  for (size_t i = 0; i < batch_dims_a; ++i) {
    batch_shape_a[i] = a_shape[i];
  }
  for (size_t i = 0; i < batch_dims_b; ++i) {
    batch_shape_b[i] = b_shape[i];
  }

  size_t a_matrix_size = a_shape[a_dims - 2] * a_shape[a_dims - 1];
  size_t b_matrix_size = b_shape[b_dims - 2] * b_shape[b_dims - 1];
  size_t out_matrix_size = a_shape[a_dims - 2] * b_shape[b_dims - 1];

  std::vector<size_t> a_batch_strides(max_batch_dims, a_matrix_size);
  std::vector<size_t> b_batch_strides(max_batch_dims, b_matrix_size);
  std::vector<size_t> out_batch_strides(max_batch_dims, out_matrix_size);

  for (int i = static_cast<int>(max_batch_dims) - 2; i >= 0; --i) {
    auto idx = static_cast<size_t>(i);
    a_batch_strides[idx] = a_batch_strides[idx + 1] * batch_shape_a[idx + 1];
    b_batch_strides[idx] = b_batch_strides[idx + 1] * batch_shape_b[idx + 1];
  }

  std::vector<size_t> output_batch_shape(max_batch_dims);
  for (size_t i = 0; i < max_batch_dims; ++i) {
    if (batch_shape_a[i] != batch_shape_b[i] && batch_shape_a[i] != 1 &&
        batch_shape_b[i] != 1) {
      throw std::runtime_error(
          "MatMul: Incompatible batch dimensions for broadcasting");
    }
    output_batch_shape[i] = std::max(batch_shape_a[i], batch_shape_b[i]);
  }

  for (int i = static_cast<int>(max_batch_dims) - 2; i >= 0; --i) {
    auto idx = static_cast<size_t>(i);
    out_batch_strides[idx] =
        out_batch_strides[idx + 1] * output_batch_shape[idx + 1];
  }

  std::vector<size_t> output_shape = output_batch_shape;
  output_shape.push_back(a_shape[a_dims - 2]);
  output_shape.push_back(b_shape[b_dims - 1]);

  size_t m = a_shape[a_dims - 2];
  size_t n = b_shape[b_dims - 1];
  size_t k = a_shape[a_dims - 1];

  size_t total_batch = 1;
  for (size_t dim : output_batch_shape) {
    total_batch *= dim;
  }

  std::vector<T> output_values(total_batch * m * n, T(0));

  for (size_t batch = 0; batch < total_batch; ++batch) {
    size_t a_batch_idx = 0;
    size_t b_batch_idx = 0;
    size_t out_batch_idx = 0;
    size_t temp_batch = batch;

    for (int i = static_cast<int>(max_batch_dims) - 1; i >= 0; --i) {
      auto idx = static_cast<size_t>(i);
      size_t dim_size = output_batch_shape[idx];
      size_t batch_idx = temp_batch % dim_size;
      temp_batch /= dim_size;

      if (batch_shape_a[idx] > 1) {
        a_batch_idx += batch_idx * a_batch_strides[idx];
      }
      if (batch_shape_b[idx] > 1) {
        b_batch_idx += batch_idx * b_batch_strides[idx];
      }
      out_batch_idx += batch_idx * out_batch_strides[idx];
    }

    size_t a_offset = a_batch_idx;
    size_t b_offset = b_batch_idx;
    size_t out_offset = out_batch_idx;

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        T sum = T(0);
        for (size_t l = 0; l < k; ++l) {
          size_t a_index = a_offset + i * k + l;
          size_t b_index = b_offset + l * n + j;
          if (a_index >= a_data->size()) {
            std::cerr << "a_idx out of bounds: " << a_index
                      << " >= " << a_data->size() << std::endl;
            throw std::runtime_error("MatMul: a index out of bounds");
          }
          if (b_index >= b_data->size()) {
            std::cerr << "b_idx out of bounds: " << b_index
                      << " >= " << b_data->size() << std::endl;
            throw std::runtime_error("MatMul: b index out of bounds");
          }
          sum += (*a_data)[a_index] * (*b_data)[b_index];
        }
        output_values[out_offset + i * n + j] = sum;
      }
    }
  }

  output = make_tensor(output_values, output_shape);
}

template void MatmulLayer::matmul_impl<float>(const Tensor&, const Tensor&,
                                              Tensor&) const;
template void MatmulLayer::matmul_impl<int>(const Tensor&, const Tensor&,
                                            Tensor&) const;

}  // namespace it_lab_ai