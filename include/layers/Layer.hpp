#pragma once
#include <algorithm>
#include <execution>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "graph/runtime_options.hpp"
#include "layers/Shape.hpp"
#include "layers/Tensor.hpp"
#include "parallel/parallel.hpp"

namespace it_lab_ai {

enum LayerType : uint8_t {
  kInput,
  kPooling,
  kNormalization,
  kDropout,
  kElementWise,
  kConvolution,
  kFullyConnected,
  kFlatten,
  kOutput,
  kConcat,
  kSplit,
  kBinaryOp,
  kReduce,
  kTranspose,
  kReshape,
  kSoftmax,
  kMatmul,
  kBatchNormalization,
  kConvRelu
};

enum ImplType : uint8_t { kDefault, kTBB, kSTL };
using ParBackend = parallel::Backend;

class Layer;

struct PostOperations {
  std::vector<std::shared_ptr<Layer>> layers;
  unsigned int count = 0;
};

class Layer {
 public:
  Layer() = default;
  explicit Layer(LayerType type) : type_(type) {}
  virtual ~Layer() = default;
  PostOperations postops;
  [[nodiscard]] int getID() const {
    return id_;
  }
  void setID(int id) {
    id_ = id;
  }
  [[nodiscard]] LayerType getName() const {
    return type_;
  }
  virtual void run(const std::vector<Tensor>& input,
                   std::vector<Tensor>& output) = 0;
  virtual void run(const std::vector<Tensor>& input,
                   std::vector<Tensor>& output,
                   [[maybe_unused]] const RuntimeOptions& options) {
    run(input, output);
  }
#ifdef ENABLE_STATISTIC_WEIGHTS
  virtual Tensor get_weights() = 0;
#endif

 protected:
  int id_ = 0;
  LayerType type_;
};

template <typename ValueType>
class LayerImpl {
 public:
  LayerImpl() = default;
  LayerImpl(const Shape& inputShape, const Shape& outputShape)
      : inputShape_(inputShape), outputShape_(outputShape) {}
  virtual ~LayerImpl() = default;
  LayerImpl(const LayerImpl& c) = default;
  LayerImpl& operator=(const LayerImpl& c) = default;
  [[nodiscard]] virtual std::vector<ValueType> run(
      const std::vector<ValueType>& input) const = 0;
  [[nodiscard]] Shape get_input_shape() const {
    return inputShape_;
  }
  [[nodiscard]] Shape get_output_shape() const {
    return outputShape_;
  }
  // weights width x height
  [[nodiscard]] std::pair<Shape, Shape> get_dims() const {
    return std::pair<Shape, Shape>(outputShape_, inputShape_);
  }

 protected:
  Shape inputShape_;
  Shape outputShape_;
};
}  // namespace it_lab_ai
