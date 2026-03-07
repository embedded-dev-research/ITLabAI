#include "graph/graph.hpp"

#include "layers/BatchNormalizationLayer.hpp"
#include "layers/BinaryOpLayer.hpp"
#include "layers/ConcatLayer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/DropOutLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/MatmulLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "layers/ReduceLayer.hpp"
#include "layers/ReshapeLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/SplitLayer.hpp"
#include "layers/Tensor.hpp"
#include "layers/TransposeLayer.hpp"
#include "layers_oneDNN/BinaryOpLayer.hpp"
#include "layers_oneDNN/ConvLayer.hpp"
#include "layers_oneDNN/EWLayer.hpp"
#include "layers_oneDNN/PoolingLayer.hpp"
#include "layers_oneDNN/ReduceLayer.hpp"

namespace it_lab_ai {

namespace {
template <typename T>
std::shared_ptr<Layer> clone_layer_checked(
    const std::shared_ptr<Layer>& layer) {
  const auto* casted = dynamic_cast<const T*>(layer.get());
  if (casted == nullptr) {
    throw std::invalid_argument("Layer type mismatch while cloning");
  }
  return std::make_shared<T>(*casted);
}
}  // namespace

void Graph::clone(Graph& result, Tensor& out,
                  const RuntimeOptions& options) const {
  result.arrayE_ = this->arrayE_;
  result.arrayV_ = this->arrayV_;
  result.BiggestSize_ = this->BiggestSize_;
  result.branch_map_ = this->branch_map_;
  result.count_used_split_distribution_ = this->count_used_split_distribution_;
  result.end_ = this->end_;
  result.inten_ = this->inten_;
  result.in_edges_ = this->in_edges_;
  result.outtenres_ = &out;
  result.outten_ = this->outten_;
  result.split_distribution_ = this->split_distribution_;
  result.start_ = this->start_;
  result.V_ = this->V_;
  result.layers_ = std::vector<std::shared_ptr<Layer>>();
  for (const auto& layer : this->layers_) {
    result.layers_.push_back(layer_based_shared_copy(layer, options));
  }
#ifdef ENABLE_STATISTIC_TENSORS
  result.tensors_ = this->tensors_;
#endif
#ifdef ENABLE_STATISTIC_TIME
  result.time_ = this->time_;
  result.time_layer_ = this->time_layer_;
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
  result.weights_ = this->weights_;
#endif
}

std::shared_ptr<Layer> layer_based_shared_copy(
    const std::shared_ptr<Layer>& layer, const RuntimeOptions& options) {
  switch (layer->getName()) {
    case it_lab_ai::kInput: {
      return clone_layer_checked<InputLayer>(layer);
    }
    case it_lab_ai::kPooling: {
      if (options.backend == Backend::kOneDnn) {
        return clone_layer_checked<PoolingLayerOneDnn>(layer);
      }
      return clone_layer_checked<PoolingLayer>(layer);
    }
    case it_lab_ai::kElementWise: {
      if (options.backend == Backend::kOneDnn) {
        return clone_layer_checked<EwLayerOneDnn>(layer);
      }
      return clone_layer_checked<EWLayer>(layer);
    }
    case it_lab_ai::kConvolution: {
      if (options.backend == Backend::kOneDnn) {
        return clone_layer_checked<ConvLayerOneDnn>(layer);
      }
      return clone_layer_checked<ConvolutionalLayer>(layer);
    }
    case it_lab_ai::kFullyConnected: {
      return clone_layer_checked<FCLayer>(layer);
    }
    case it_lab_ai::kFlatten: {
      return clone_layer_checked<FlattenLayer>(layer);
    }
    case it_lab_ai::kConcat: {
      return clone_layer_checked<ConcatLayer>(layer);
    }
    case it_lab_ai::kDropout: {
      return clone_layer_checked<DropOutLayer>(layer);
    }
    case it_lab_ai::kSplit: {
      return clone_layer_checked<SplitLayer>(layer);
    }
    case it_lab_ai::kBinaryOp: {
      if (options.backend == Backend::kOneDnn) {
        return clone_layer_checked<BinaryOpLayerOneDnn>(layer);
      }
      return clone_layer_checked<BinaryOpLayer>(layer);
    }
    case it_lab_ai::kTranspose: {
      return clone_layer_checked<TransposeLayer>(layer);
    }
    case it_lab_ai::kMatmul: {
      return clone_layer_checked<MatmulLayer>(layer);
    }
    case it_lab_ai::kReshape: {
      return clone_layer_checked<ReshapeLayer>(layer);
    }
    case it_lab_ai::kSoftmax: {
      return clone_layer_checked<SoftmaxLayer>(layer);
    }
    case it_lab_ai::kReduce: {
      if (options.backend == Backend::kOneDnn) {
        return clone_layer_checked<ReduceLayerOneDnn>(layer);
      }
      return clone_layer_checked<ReduceLayer>(layer);
    }
    case it_lab_ai::kBatchNormalization: {
      return clone_layer_checked<BatchNormalizationLayer>(layer);
    }
    case it_lab_ai::kOutput: {
      return clone_layer_checked<OutputLayer>(layer);
    }
    default: {
      throw std::invalid_argument("No such layer type");
    }
  }
}
}  // namespace it_lab_ai
