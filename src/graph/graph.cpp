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
  for (int i = 0; i < this->layers_.size(); i++) {
    result.layers_.push_back(
        layer_based_shared_copy(this->layers_[i], options));
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
      InputLayer* tmp_layer =
          new InputLayer(*dynamic_cast<InputLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kPooling: {
      if (options.backend == Backend::kOneDnn) {
        PoolingLayerOneDnn* tmp_layer = new PoolingLayerOneDnn(
            *dynamic_cast<PoolingLayerOneDnn*>(layer.get()));
        return std::shared_ptr<Layer>(tmp_layer);
      }
      PoolingLayer* tmp_layer =
          new PoolingLayer(*dynamic_cast<PoolingLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kElementWise: {
      if (options.backend == Backend::kOneDnn) {
        EwLayerOneDnn* tmp_layer =
            new EwLayerOneDnn(*dynamic_cast<EwLayerOneDnn*>(layer.get()));
        return std::shared_ptr<Layer>(tmp_layer);
      }
      EWLayer* tmp_layer = new EWLayer(*dynamic_cast<EWLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kConvolution: {
      if (options.backend == Backend::kOneDnn) {
        ConvLayerOneDnn* tmp_layer =
            new ConvLayerOneDnn(*dynamic_cast<ConvLayerOneDnn*>(layer.get()));
        return std::shared_ptr<Layer>(tmp_layer);
      }
      ConvolutionalLayer* tmp_layer = new ConvolutionalLayer(
          *dynamic_cast<ConvolutionalLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kFullyConnected: {
      FCLayer* tmp_layer = new FCLayer(*dynamic_cast<FCLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kFlatten: {
      FlattenLayer* tmp_layer =
          new FlattenLayer(*dynamic_cast<FlattenLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kConcat: {
      ConcatLayer* tmp_layer =
          new ConcatLayer(*dynamic_cast<ConcatLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kDropout: {
      DropOutLayer* tmp_layer =
          new DropOutLayer(*dynamic_cast<DropOutLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kSplit: {
      SplitLayer* tmp_layer =
          new SplitLayer(*dynamic_cast<SplitLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kBinaryOp: {
      if (options.backend == Backend::kOneDnn) {
        BinaryOpLayerOneDnn* tmp_layer = new BinaryOpLayerOneDnn(
            *dynamic_cast<BinaryOpLayerOneDnn*>(layer.get()));
        return std::shared_ptr<Layer>(tmp_layer);
      }
      BinaryOpLayer* tmp_layer =
          new BinaryOpLayer(*dynamic_cast<BinaryOpLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kTranspose: {
      TransposeLayer* tmp_layer =
          new TransposeLayer(*dynamic_cast<TransposeLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kMatmul: {
      MatmulLayer* tmp_layer =
          new MatmulLayer(*dynamic_cast<MatmulLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kReshape: {
      ReshapeLayer* tmp_layer =
          new ReshapeLayer(*dynamic_cast<ReshapeLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kSoftmax: {
      SoftmaxLayer* tmp_layer =
          new SoftmaxLayer(*dynamic_cast<SoftmaxLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kReduce: {
      if (options.backend == Backend::kOneDnn) {
        ReduceLayerOneDnn* tmp_layer = new ReduceLayerOneDnn(
            *dynamic_cast<ReduceLayerOneDnn*>(layer.get()));
        return std::shared_ptr<Layer>(tmp_layer);
      }
      ReduceLayer* tmp_layer =
          new ReduceLayer(*dynamic_cast<ReduceLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    case it_lab_ai::kBatchNormalization: {
      BatchNormalizationLayer* tmp_layer = new BatchNormalizationLayer(
          *dynamic_cast<BatchNormalizationLayer*>(layer.get()));
      return std::shared_ptr<Layer>(tmp_layer);
    }
    default: {
      throw std::invalid_argument("No such layer type");
    }
  }
}
}  // namespace it_lab_ai
