#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "layers_oneDNN/ReduceLayer.hpp"

namespace it_lab_ai {

ReduceLayerOneDnn::ReduceLayerOneDnn(ReduceLayer::Operation op,
                                     int64_t keepdims,
                                     const std::vector<int64_t>& axes)
    : Layer(kReduce), op_(op), keepdims_(keepdims), axes_(axes) {}

void ReduceLayerOneDnn::run(const std::vector<Tensor>& input,
                            std::vector<Tensor>& output) {
  validate_input(input);

  const Tensor& in = input[0];
  Type type = in.get_type();

  std::vector<int64_t> normalized_axes = axes_;
  normalize_axes(in.get_shape(), normalized_axes);

  bool need_reinit = !initialized_ || last_type_ != type ||
                     last_input_shape_ != in.get_shape() ||
                     // Проверяем, изменились ли нормализованные оси
                     normalized_axes != axes_;

  if (need_reinit) {
    initialize_onednn(in);
  }

  output.resize(1);

  // Специальная обработка для операции mean
  if (op_ == ReduceLayer::Operation::kMean) {
    compute_mean(in, output[0]);
    return;
  }

  if (type == Type::kFloat) {
    const auto& src_data = *in.as<float>();
    std::vector<float> dst_data(output_shape_.count());

    dnnl::memory src_mem(src_md_, *engine_,
                         const_cast<float*>(src_data.data()));
    dnnl::memory dst_mem(dst_md_, *engine_, dst_data.data());

    reduction_prim_->execute(
        *stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});

    stream_->wait();
    output[0] = make_tensor(dst_data, output_shape_);
  } else if (type == Type::kInt) {
    // oneDNN reduction не поддерживает целочисленные типы для всех операций
    // Используем fallback на CPU реализацию для целых чисел
    const auto& src_data = *in.as<int>();
    std::vector<int> dst_data(output_shape_.count());

    // Для целых чисел используем собственную реализацию
    const auto& input_shape = in.get_shape();

    // Инициализируем выходные значения
    switch (op_) {
      case ReduceLayer::Operation::kSum:
        std::fill(dst_data.begin(), dst_data.end(), 0);
        break;
      case ReduceLayer::Operation::kMult:
        std::fill(dst_data.begin(), dst_data.end(), 1);
        break;
      case ReduceLayer::Operation::kMax:
        std::fill(dst_data.begin(), dst_data.end(),
                  std::numeric_limits<int>::lowest());
        break;
      case ReduceLayer::Operation::kMin:
        std::fill(dst_data.begin(), dst_data.end(),
                  std::numeric_limits<int>::max());
        break;
      case ReduceLayer::Operation::kMean:
        // Обработано отдельно
        break;
    }

    // Вычисляем индексы
    const size_t input_rank = input_shape.dims();
    std::vector<size_t> input_coords(input_rank, 0);

    for (size_t in_idx = 0; in_idx < src_data.size(); ++in_idx) {
      // Вычисляем выходные координаты
      std::vector<size_t> output_coords;
      if (keepdims_) {
        output_coords.resize(input_rank, 0);
        for (size_t i = 0; i < input_rank; ++i) {
          if (std::find(normalized_axes.begin(), normalized_axes.end(),
                        static_cast<int64_t>(i)) == normalized_axes.end()) {
            output_coords[i] = input_coords[i];
          }
        }
      } else {
        for (size_t i = 0; i < input_rank; ++i) {
          if (std::find(normalized_axes.begin(), normalized_axes.end(),
                        static_cast<int64_t>(i)) == normalized_axes.end()) {
            output_coords.push_back(input_coords[i]);
          }
        }
      }

      // Вычисляем выходной индекс
      size_t out_idx = 0;
      size_t stride = 1;
      for (size_t i = output_coords.size(); i-- > 0;) {
        out_idx += output_coords[i] * stride;
        stride *= output_shape_[i];
      }

      // Применяем операцию
      switch (op_) {
        case ReduceLayer::Operation::kSum:
          dst_data[out_idx] += src_data[in_idx];
          break;
        case ReduceLayer::Operation::kMult:
          dst_data[out_idx] *= src_data[in_idx];
          break;
        case ReduceLayer::Operation::kMax:
          if (src_data[in_idx] > dst_data[out_idx]) {
            dst_data[out_idx] = src_data[in_idx];
          }
          break;
        case ReduceLayer::Operation::kMin:
          if (src_data[in_idx] < dst_data[out_idx]) {
            dst_data[out_idx] = src_data[in_idx];
          }
          break;
        case ReduceLayer::Operation::kMean:
          // Не должно сюда попадать
          break;
      }

      // Обновляем входные координаты
      for (size_t i = input_rank; i-- > 0;) {
        ++input_coords[i];
        if (input_coords[i] < input_shape[i]) break;
        input_coords[i] = 0;
      }
    }

    output[0] = make_tensor(dst_data, output_shape_);
  }
}

void ReduceLayerOneDnn::compute_mean(const Tensor& input, Tensor& output) {
  Type type = input.get_type();

  if (type == Type::kFloat) {
    // Вычисляем сумму
    const auto& src_data = *input.as<float>();
    std::vector<float> sum_data(output_shape_.count());

    dnnl::memory src_mem(src_md_, *engine_,
                         const_cast<float*>(src_data.data()));
    dnnl::memory sum_mem(dst_md_, *engine_, sum_data.data());

    // Выполняем операцию суммы
    reduction_prim_->execute(
        *stream_, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, sum_mem}});

    stream_->wait();

    // Вычисляем количество элементов для усреднения
    const Shape& input_shape = input.get_shape();
    std::vector<int64_t> normalized_axes = axes_;
    normalize_axes(input_shape, normalized_axes);

    size_t reduction_size = 1;
    for (int64_t axis : normalized_axes) {
      reduction_size *= input_shape[axis];
    }

    float scale = 1.0f / static_cast<float>(reduction_size);

    // Делим на количество элементов
    std::vector<float> mean_data(sum_data.size());
    for (size_t i = 0; i < sum_data.size(); ++i) {
      mean_data[i] = sum_data[i] * scale;
    }

    output = make_tensor(mean_data, output_shape_);

  } else if (type == Type::kInt) {
    // Для целых чисел используем целочисленное деление
    const auto& src_data = *input.as<int>();
    std::vector<int> sum_data(output_shape_.count(), 0);

    const Shape& input_shape = input.get_shape();
    std::vector<int64_t> normalized_axes = axes_;
    normalize_axes(input_shape, normalized_axes);

    // Вычисляем сумму
    size_t input_rank = input_shape.dims();
    std::vector<size_t> input_coords(input_rank, 0);

    for (size_t in_idx = 0; in_idx < src_data.size(); ++in_idx) {
      // Вычисляем выходные координаты
      std::vector<size_t> output_coords;
      if (keepdims_) {
        output_coords.resize(input_rank, 0);
        for (size_t i = 0; i < input_rank; ++i) {
          if (std::find(normalized_axes.begin(), normalized_axes.end(),
                        static_cast<int64_t>(i)) == normalized_axes.end()) {
            output_coords[i] = input_coords[i];
          }
        }
      } else {
        for (size_t i = 0; i < input_rank; ++i) {
          if (std::find(normalized_axes.begin(), normalized_axes.end(),
                        static_cast<int64_t>(i)) == normalized_axes.end()) {
            output_coords.push_back(input_coords[i]);
          }
        }
      }

      // Вычисляем выходной индекс
      size_t out_idx = 0;
      size_t stride = 1;
      for (size_t i = output_coords.size(); i-- > 0;) {
        out_idx += output_coords[i] * stride;
        stride *= output_shape_[i];
      }

      sum_data[out_idx] += src_data[in_idx];

      // Обновляем входные координаты
      for (size_t i = input_rank; i-- > 0;) {
        ++input_coords[i];
        if (input_coords[i] < input_shape[i]) break;
        input_coords[i] = 0;
      }
    }

    // Делим на количество элементов
    size_t reduction_size = 1;
    for (int64_t axis : normalized_axes) {
      reduction_size *= input_shape[axis];
    }

    std::vector<int> mean_data(sum_data.size());
    for (size_t i = 0; i < sum_data.size(); ++i) {
      mean_data[i] = sum_data[i] / static_cast<int>(reduction_size);
    }

    output = make_tensor(mean_data, output_shape_);
  }
}

void ReduceLayerOneDnn::validate_input(const std::vector<Tensor>& input) {
  if (input.size() != 1) {
    throw std::runtime_error(
        "ReduceLayerOneDnn: Expected exactly 1 input tensor");
  }

  const auto& shape = input[0].get_shape();
  if (shape.dims() == 0) {
    throw std::runtime_error("ReduceLayerOneDnn: Scalar input not supported");
  }
}

Shape ReduceLayerOneDnn::calculate_output_shape(
    const Shape& input_shape, const std::vector<int64_t>& axes) const {
  if (input_shape.dims() == 0) {
    return Shape({});
  }

  std::vector<size_t> new_dims;

  if (keepdims_) {
    new_dims.resize(input_shape.dims(), 1);
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.dims()); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
      if (!is_axis) {
        new_dims[i] = input_shape[i];
      }
    }
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(input_shape.dims()); ++i) {
      bool is_axis = std::find(axes.begin(), axes.end(), i) != axes.end();
      if (!is_axis) {
        new_dims.push_back(input_shape[i]);
      }
    }
    if (new_dims.empty()) {
      new_dims.push_back(1);
    }
  }

  return Shape(new_dims);
}

void ReduceLayerOneDnn::normalize_axes(const Shape& input_shape,
                                       std::vector<int64_t>& axes) {
  const auto rank = static_cast<int64_t>(input_shape.dims());

  if (rank == 0) {
    if (!axes.empty()) {
      throw std::runtime_error(
          "ReduceLayerOneDnn: Axis specified for scalar input");
    }
    return;
  }

  if (axes.empty()) {
    axes.resize(rank);
    std::iota(axes.begin(), axes.end(), 0);
    return;
  }

  for (auto& axis : axes) {
    if (axis < -rank || axis >= rank) {
      throw std::runtime_error(
          "ReduceLayerOneDnn: Axis out of range. Valid range is [-" +
          std::to_string(rank) + ", " + std::to_string(rank - 1) + "]");
    }

    if (axis < 0) {
      axis += rank;
    }
  }

  std::sort(axes.begin(), axes.end());
  axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
}

void ReduceLayerOneDnn::initialize_onednn(const Tensor& input) {
  engine_ = std::make_unique<dnnl::engine>(dnnl::engine::kind::cpu, 0);
  stream_ = std::make_unique<dnnl::stream>(*engine_);

  const Shape& input_shape = input.get_shape();
  std::vector<int64_t> normalized_axes = axes_;
  normalize_axes(input_shape, normalized_axes);

  output_shape_ = calculate_output_shape(input_shape, normalized_axes);

  auto dnnl_type = get_dnnl_data_type(input.get_type());

  auto src_dims = shape_to_dims(input_shape);
  auto dst_dims = shape_to_dims(output_shape_);

  size_t ndims = input_shape.dims();
  auto format = pick_format(ndims);

  src_md_ = dnnl::memory::desc(src_dims, dnnl_type, format);
  dst_md_ = dnnl::memory::desc(dst_dims, dnnl_type, format);

  try {
    // Для операций, которые поддерживает oneDNN
    if (op_ != ReduceLayer::Operation::kMult) {  // oneDNN не поддерживает
                                                 // reduction multiplication
      float p = 0.0f;    // Параметр для алгоритмов
      float eps = 0.0f;  // Эпсилон

      auto reduction_pd = dnnl::reduction::primitive_desc(
          *engine_, get_dnnl_algorithm(op_), src_md_, dst_md_, p, eps);

      reduction_prim_ = std::make_unique<dnnl::reduction>(reduction_pd);
    }

  } catch (const dnnl::error& e) {
    std::cerr << "Error creating reduction primitive: " << e.what() << '\n';
    throw std::runtime_error("Failed to create reduction primitive: " +
                             std::string(e.what()));
  }

  last_input_shape_ = input_shape;
  last_type_ = input.get_type();
  initialized_ = true;
}

dnnl::memory::data_type ReduceLayerOneDnn::get_dnnl_data_type(Type type) {
  switch (type) {
    case Type::kFloat:
      return dnnl::memory::data_type::f32;
    case Type::kInt:
      return dnnl::memory::data_type::s32;
    default:
      throw std::runtime_error("Unsupported data type for oneDNN");
  }
}

dnnl::algorithm ReduceLayerOneDnn::get_dnnl_algorithm(
    ReduceLayer::Operation op) {
  switch (op) {
    case ReduceLayer::Operation::kSum:
      return dnnl::algorithm::reduction_sum;
    case ReduceLayer::Operation::kMean:
      return dnnl::algorithm::reduction_sum;  // Для mean сначала вычисляем
                                              // сумму
    case ReduceLayer::Operation::kMax:
      return dnnl::algorithm::reduction_max;
    case ReduceLayer::Operation::kMin:
      return dnnl::algorithm::reduction_min;
    case ReduceLayer::Operation::kMult:
      // oneDNN не поддерживает reduction multiplication
      throw std::runtime_error(
          "Multiplication reduction not supported by oneDNN");
    default:
      throw std::invalid_argument("Unsupported reduction operation for oneDNN");
  }
}

dnnl::memory::format_tag ReduceLayerOneDnn::pick_format(size_t ndims) {
  switch (ndims) {
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

std::vector<dnnl::memory::dim> ReduceLayerOneDnn::shape_to_dims(
    const Shape& shape) {
  std::vector<dnnl::memory::dim> dims;
  for (size_t i = 0; i < shape.dims(); ++i) {
    dims.push_back(static_cast<dnnl::memory::dim>(shape.at(i)));
  }
  return dims;
}

std::vector<dnnl::memory::dim> ReduceLayerOneDnn::get_dnnl_axes(
    const std::vector<int64_t>& axes) {
  std::vector<dnnl::memory::dim> dnnl_axes;
  for (int64_t axis : axes) {
    dnnl_axes.push_back(static_cast<dnnl::memory::dim>(axis));
  }
  return dnnl_axes;
}

}  // namespace it_lab_ai