#include "layers_oneDNN/BinaryOpLayer.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace it_lab_ai {

void BinaryOpLayerOneDnn::run(const std::vector<Tensor>& input,
                              std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& a = input[0];
  const Tensor& b = input[1];
  Type type = a.get_type();

  bool need_reinit = !initialized_ || last_type_ != type ||
                     last_shape_a_ != a.get_shape() ||
                     last_shape_b_ != b.get_shape();

  if (need_reinit) {
    initialize_onednn(a, b);
  }

  output.resize(1);
  output_shape_ = calculate_output_shape(a.get_shape(), b.get_shape());

  if (type == Type::kFloat) {
    const auto& src0_data = *a.as<float>();
    const auto& src1_data = *b.as<float>();
    std::vector<float> dst_data(output_shape_.count());

    dnnl::memory src0_mem(src0_md_, *engine_,
                          const_cast<float*>(src0_data.data()));
    dnnl::memory src1_mem(src1_md_, *engine_,
                          const_cast<float*>(src1_data.data()));
    dnnl::memory dst_mem(dst_md_, *engine_, dst_data.data());

    binary_prim_->execute(*stream_, {{DNNL_ARG_SRC_0, src0_mem},
                                     {DNNL_ARG_SRC_1, src1_mem},
                                     {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst_data, output_shape_);
  } else if (type == Type::kInt) {
    const auto& src0_data = *a.as<int>();
    const auto& src1_data = *b.as<int>();
    std::vector<int> dst_data(output_shape_.count());

    dnnl::memory src0_mem(src0_md_, *engine_,
                          const_cast<int*>(src0_data.data()));
    dnnl::memory src1_mem(src1_md_, *engine_,
                          const_cast<int*>(src1_data.data()));
    dnnl::memory dst_mem(dst_md_, *engine_, dst_data.data());

    binary_prim_->execute(*stream_, {{DNNL_ARG_SRC_0, src0_mem},
                                     {DNNL_ARG_SRC_1, src1_mem},
                                     {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst_data, output_shape_);
  }
}

void BinaryOpLayerOneDnn::validate_input(const std::vector<Tensor>& input) {
  if (input.size() != 2) {
    throw std::runtime_error(
        "BinaryOpLayerOneDnn: Expected exactly 2 input tensors");
  }

  if (input[0].get_type() != input[1].get_type()) {
    throw std::runtime_error(
        "BinaryOpLayerOneDnn: Input tensors must have the same type");
  }

  const Shape& shape_a = input[0].get_shape();
  const Shape& shape_b = input[1].get_shape();

  if (!can_broadcast(shape_a, shape_b)) {
    throw std::runtime_error(
        "BinaryOpLayerOneDnn: Incompatible shapes for broadcasting");
  }
}

Shape BinaryOpLayerOneDnn::calculate_output_shape(const Shape& shape_a,
                                                  const Shape& shape_b) {
  size_t dims_a = shape_a.dims();
  size_t dims_b = shape_b.dims();
  size_t max_dims = std::max(dims_a, dims_b);
  Shape result(max_dims);

  for (size_t i = 0; i < max_dims; ++i) {
    size_t idx_a = dims_a - i - 1;
    size_t idx_b = dims_b - i - 1;
    size_t idx_result = max_dims - i - 1;

    size_t dim_a = (i < dims_a) ? shape_a[idx_a] : 1;
    size_t dim_b = (i < dims_b) ? shape_b[idx_b] : 1;

    if ((dim_a != dim_b) && (dim_a != 1) && (dim_b != 1)) {
      throw std::runtime_error("BinaryOpLayerOneDnn: Incompatible dimensions");
    }
    result[idx_result] = std::max(dim_a, dim_b);
  }

  return result;
}

bool BinaryOpLayerOneDnn::can_broadcast(const Shape& shape_a,
                                        const Shape& shape_b) {
  size_t dims_a = shape_a.dims();
  size_t dims_b = shape_b.dims();
  size_t max_dims = std::max(dims_a, dims_b);

  for (size_t i = 0; i < max_dims; ++i) {
    size_t idx_a = dims_a - i - 1;
    size_t idx_b = dims_b - i - 1;

    size_t dim_a = (i < dims_a) ? shape_a[idx_a] : 1;
    size_t dim_b = (i < dims_b) ? shape_b[idx_b] : 1;

    if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
      return false;
    }
  }

  return true;
}

void BinaryOpLayerOneDnn::initialize_onednn(const Tensor& A, const Tensor& B) {
  engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
  stream_ = std::make_unique<dnnl::stream>(*engine_);

  const Shape& shape_a = A.get_shape();
  const Shape& shape_b = B.get_shape();
  output_shape_ = calculate_output_shape(shape_a, shape_b);

  auto dnnl_type = get_dnnl_data_type(A.get_type());

  auto dims_a = shape_to_dims(shape_a);
  auto dims_b = shape_to_dims(shape_b);
  auto dims_output = shape_to_dims(output_shape_);

  size_t ndims = output_shape_.dims();
  auto format = pick_format(ndims);

  src0_md_ = dnnl::memory::desc(dims_a, dnnl_type, format);
  src1_md_ = dnnl::memory::desc(dims_b, dnnl_type, format);
  dst_md_ = dnnl::memory::desc(dims_output, dnnl_type, format);

  try {
    auto binary_pd = dnnl::binary::primitive_desc(
        *engine_, get_dnnl_algorithm(op_), src0_md_, src1_md_, dst_md_);

    binary_prim_ = std::make_unique<dnnl::binary>(binary_pd);
  } catch (const dnnl::error& e) {
    std::cerr << "Error creating binary primitive: " << e.what() << '\n';
    throw std::runtime_error("Failed to create binary primitive: " +
                             std::string(e.what()));
  }

  last_shape_a_ = shape_a;
  last_shape_b_ = shape_b;
  last_type_ = A.get_type();
  initialized_ = true;
}

dnnl::memory::data_type BinaryOpLayerOneDnn::get_dnnl_data_type(Type type) {
  switch (type) {
    case Type::kFloat:
      return dnnl::memory::data_type::f32;
    case Type::kInt:
      return dnnl::memory::data_type::s32;
    default:
      throw std::runtime_error("Unsupported data type for oneDNN");
  }
}

dnnl::algorithm BinaryOpLayerOneDnn::get_dnnl_algorithm(
    BinaryOpLayer::Operation op) {
  switch (op) {
    case BinaryOpLayer::Operation::kAdd:
      return dnnl::algorithm::binary_add;
    case BinaryOpLayer::Operation::kMul:
      return dnnl::algorithm::binary_mul;
    default:
      throw std::invalid_argument("Unsupported binary operation for oneDNN");
  }
}

dnnl::memory::format_tag BinaryOpLayerOneDnn::pick_format(size_t ndims) {
  switch (ndims) {
    case 0:
    case 1:
      return dnnl::memory::format_tag::a;
    case 2:
      return dnnl::memory::format_tag::ab;
    case 3:
      return dnnl::memory::format_tag::abc;
    case 4:
      return dnnl::memory::format_tag::abcd;
    case 5:
      return dnnl::memory::format_tag::abcde;
    default:
      return dnnl::memory::format_tag::any;
  }
}

std::vector<dnnl::memory::dim> BinaryOpLayerOneDnn::shape_to_dims(
    const Shape& shape) {
  std::vector<dnnl::memory::dim> dims;
  for (size_t i = 0; i < shape.dims(); ++i) {
    dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
  }

  if (dims.empty()) {
    dims.push_back(1);
  }

  return dims;
}

}  // namespace it_lab_ai
