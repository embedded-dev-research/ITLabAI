#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "layers/Tensor.hpp"
#include "graph/graph.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"

using json = nlohmann::json;
using namespace itlab_2023;

json read_json(const std::string& filename);
void extract_values_without_bias(const json& j, std::vector<float>& values);
void extract_values_from_json(const json& j, std::vector<float>& values);
void parse_json_shape(const json& j, std::vector<size_t>& shape,
                      size_t dim = 0);
Tensor create_tensor_from_json(const json& j, Type type);
void extract_bias_from_json(const json& j, std::vector<float>& bias);

Graph make_graph_with_weights(Tensor pic) {
  Graph graph(15);

  Tensor input = pic;
  InputLayer a1(kNhwc, kNchw, 1, 2);
  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);
  Tensor kernel;
  for (auto& layer : model_data.items()) {
    if (layer.key() == "layer_conv1")
      kernel = create_tensor_from_json(layer.value(), Type::kFloat);
  }
  ConvolutionalLayer a2(1, 2, 1, kernel);

  InputLayer a1(kNhwc, kNchw, 1, 2);
  std::vector<float> kernelvec = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  Shape sh2({3, 3});
  Tensor kernel = make_tensor(kernelvec, sh2);
  ConvolutionalLayer a2(1, 0, 0, kernel);
  Shape poolshape = {2, 2};
  EWLayer a3("linear", 2.0F, 3.0F);
  PoolingLayer a4(poolshape, "average");
  FCLayer a6;
  OutputLayer a5;
  graph.setInput(a1, input);
  graph.makeConnection(a1, a2);
  graph.makeConnection(a2, a3);
  graph.makeConnection(a3, a4);
  graph.makeConnection(a4, a5);
  graph.makeConnection(a5, a6);
  graph.setOutput(a5, output);
}