#pragma once
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/Shape.hpp"
#include "layers/Tensor.hpp"
#include "oneapi/tbb.h"

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
  kBatchNormalization
};

enum ImplType : uint8_t { kDefault, kTBB, kSTL };

class Layer;

struct PostOperations {
  std::vector<Layer*> layers;
  unsigned int count = 0;
};

class Layer {
 public:
  Layer() = default;
  Layer(LayerType type) : type_(type) {}
  virtual ~Layer() = default;
  PostOperations postops;
  int getID() const { return id_; }
  void setID(int id) { id_ = id; }
  LayerType getName() const { return type_; }
  virtual void run(const std::vector<Tensor>& input,
                   std::vector<Tensor>& output) = 0;
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
  LayerImpl(const LayerImpl& c) = default;
  LayerImpl& operator=(const LayerImpl& c) = default;
  virtual std::vector<ValueType> run(
      const std::vector<ValueType>& input) const = 0;
  Shape get_input_shape() const { return inputShape_; }
  Shape get_output_shape() const { return outputShape_; }
  // weights width x height
  std::pair<Shape, Shape> get_dims() const {
    return std::pair<Shape, Shape>(outputShape_, inputShape_);
  }

 protected:
  Shape inputShape_;
  Shape outputShape_;
};

}  // namespace it_lab_ai
