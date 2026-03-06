#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Weights_Reader/reader_weights.hpp"
#include "graph/graph.hpp"
#include "graph/runtime_options.hpp"
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
#include "layers_oneDNN/ConcatLayer.hpp"
#include "layers_oneDNN/ConvLayer.hpp"
#include "layers_oneDNN/EWLayer.hpp"
#include "layers_oneDNN/PoolingLayer.hpp"
#include "layers_oneDNN/ReduceLayer.hpp"

extern std::unordered_map<std::string, std::string> model_paths;

struct ParseResult {
  std::vector<std::shared_ptr<it_lab_ai::Layer>> layers;
  std::unordered_map<std::string, std::shared_ptr<it_lab_ai::Layer>>
      name_to_layer;
  std::unordered_map<std::string, std::vector<std::string>> connections;
  std::unordered_map<std::string, std::vector<std::string>> concat_connections;
  std::unordered_map<std::string, std::vector<int>> concat_orders;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      concat_connected_inputs;
  std::unordered_map<std::string, std::shared_ptr<it_lab_ai::SplitLayer>>
      split_layers;
  std::unordered_map<std::string, int> split_name_to_index;
  std::vector<std::vector<std::pair<int, int>>> split_distribution;
  std::unordered_map<std::string, int> original_ids;
};

void build_graph(it_lab_ai::Graph& graph, it_lab_ai::Tensor& input,
                 it_lab_ai::Tensor& output, const std::string& json_path,
                 it_lab_ai::RuntimeOptions options, bool comments);
void build_graph_linear(it_lab_ai::Graph& graph, it_lab_ai::Tensor& input,
                        it_lab_ai::Tensor& output,
                        it_lab_ai::RuntimeOptions options, bool comments);
std::unordered_map<int, std::string> load_class_names(
    const std::string& filename);

ParseResult parse_json_model(it_lab_ai::RuntimeOptions options,
                             const std::string& json_path, bool comments);

std::vector<int> get_input_shape_from_json(const std::string& json_path);
std::vector<float> process_model_output(const std::vector<float>& output,
                                        const std::string& model_name);
it_lab_ai::Tensor prepare_image(const cv::Mat& image,
                                const std::vector<int>& input_shape,
                                const std::string& model_name = "");
it_lab_ai::Tensor prepare_mnist_image(const cv::Mat& image);

void print_time_stats(it_lab_ai::Graph& graph);
namespace it_lab_ai {
class LayerFactory {
 public:
  static std::shared_ptr<Layer> createEwLayer(const std::string& function,
                                              const RuntimeOptions& options,
                                              float alpha = 1.0F,
                                              float beta = 0.0F) {
    if (options.backend == Backend::kOneDnn &&
        EwLayerOneDnn::is_function_supported(function)) {
      return std::make_shared<EwLayerOneDnn>(function, alpha, beta);
    }
    return std::make_shared<EWLayer>(function, alpha, beta);
  }

  static std::shared_ptr<Layer> createConvLayer(
      const RuntimeOptions& options, size_t step, size_t pads, size_t dilations,
      const Tensor& kernel, const Tensor& bias = Tensor(), size_t group = 1,
      bool useLegacyImpl = false) {
    if (options.backend == Backend::kOneDnn) {
      return std::make_shared<ConvLayerOneDnn>(step, pads, dilations, kernel,
                                               bias, group, useLegacyImpl);
    }
    return std::make_shared<ConvolutionalLayer>(step, pads, dilations, kernel,
                                                bias, group, useLegacyImpl);
  }

  static std::shared_ptr<Layer> createBinaryLayer(
      const it_lab_ai::BinaryOpLayer::Operation op,
      const RuntimeOptions& options) {
    if (options.backend == Backend::kOneDnn) {
      return std::make_shared<it_lab_ai::BinaryOpLayerOneDnn>(op);
    }
    return std::make_shared<it_lab_ai::BinaryOpLayer>(op);
  }

  static std::shared_ptr<Layer> createReduceLayer(
      ReduceLayer::Operation op, int64_t keepdims,
      const std::vector<int64_t>& axes, const RuntimeOptions& options) {
    if (options.backend == Backend::kOneDnn) {
      return std::make_shared<ReduceLayerOneDnn>(op, keepdims, axes);
    }
    return std::make_shared<ReduceLayer>(op, keepdims, axes);
  }

  static std::shared_ptr<Layer> createPoolingLayer(
      const std::string& PoolType, const Shape& shape,
      const RuntimeOptions& options, const Shape& strides = {2, 2},
      const Shape& pads = {0, 0, 0, 0}, const Shape& dilations = {1, 1},
      bool ceil_mode = false) {
    if (options.backend == Backend::kOneDnn) {
      return std::make_shared<PoolingLayerOneDnn>(
          shape, strides, pads, dilations, ceil_mode, PoolType);
    }
    return std::make_shared<PoolingLayer>(shape, strides, pads, dilations,
                                          ceil_mode, PoolType);
  }

  static std::shared_ptr<Layer> createConcatLayer(
      int64_t axis, const RuntimeOptions& options) {
    if (options.backend == Backend::kOneDnn) {
      return std::make_shared<ConcatLayerOneDnn>(axis);
    }
    return std::make_shared<ConcatLayer>(axis);
  }
};

}  // namespace it_lab_ai
