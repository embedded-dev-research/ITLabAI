#include <algorithm>
#include <random>
#include <vector>

#include "graph/graph.hpp"
#include "graph_transformations/graph_transformations.hpp"
#include "gtest/gtest.h"
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
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

TEST(graph, test_new_setInput) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.addSingleLayer(inputLayer);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_NO_THROW(graph.setInput(input));
}

TEST(graph, test_new_setOutput) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.addSingleLayer(inputLayer);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_NO_THROW(graph.setOutput(output));
}

TEST(graph, test_deep_copy) {
  Graph graph;
  Graph graph2;
  Graph graph_c;
  Graph graph2_c;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  auto lay1 = std::make_shared<InputLayer>();
  Shape sh = {2, 2};
  auto lay2 = std::make_shared<PoolingLayer>(sh, "average");
  auto lay2_alt = std::make_shared<PoolingLayerOneDnn>(sh);
  auto lay3 = std::make_shared<EWLayer>();
  auto lay3_alt = std::make_shared<EwLayerOneDnn>();
  auto lay4 = std::make_shared<ConvolutionalLayer>();
  auto lay4_alt = std::make_shared<ConvLayerOneDnn>();
  auto lay5 = std::make_shared<FCLayer>();
  auto lay6 = std::make_shared<FlattenLayer>();
  auto lay7 = std::make_shared<ConcatLayer>();
  auto lay8 = std::make_shared<DropOutLayer>();
  auto lay9 = std::make_shared<SplitLayer>(0, 2);
  auto lay10 = std::make_shared<BinaryOpLayer>();
  auto lay10_alt = std::make_shared<BinaryOpLayerOneDnn>();
  auto lay11 = std::make_shared<TransposeLayer>();
  auto lay12 = std::make_shared<MatmulLayer>();
  auto lay13 = std::make_shared<ReshapeLayer>();
  auto lay14 = std::make_shared<SoftmaxLayer>();
  auto lay15 = std::make_shared<ReduceLayer>();
  auto lay15_alt = std::make_shared<ReduceLayerOneDnn>();
  Tensor scale = make_tensor<float>({1.0f}, {1});
  Tensor bias = make_tensor<float>({0.0f}, {1});
  Tensor mean = make_tensor<float>({0.0f}, {1});
  Tensor var = make_tensor<float>({1.0f}, {1});
  auto lay16 =
      std::make_shared<BatchNormalizationLayer>(scale, bias, mean, var);
  auto lay17 = std::make_shared<OutputLayer>();
  graph.setInput(lay1, input);
  graph2.setInput(lay1, input);
  graph.makeConnection(lay1, lay2);
  graph2.makeConnection(lay1, lay2_alt);
  graph.makeConnection(lay1, lay3);
  graph2.makeConnection(lay1, lay3_alt);
  graph.makeConnection(lay2, lay4);
  graph2.makeConnection(lay2_alt, lay4_alt);
  graph.makeConnection(lay2, lay5);
  graph2.makeConnection(lay2_alt, lay5);
  graph.makeConnection(lay3, lay6);
  graph2.makeConnection(lay3_alt, lay6);
  graph.makeConnection(lay3, lay7);
  graph2.makeConnection(lay3_alt, lay7);
  graph.makeConnection(lay4, lay8);
  graph2.makeConnection(lay4_alt, lay8);
  graph.makeConnection(lay4, lay9);
  graph2.makeConnection(lay4_alt, lay9);
  graph.makeConnection(lay5, lay10);
  graph2.makeConnection(lay5, lay10_alt);
  graph.makeConnection(lay5, lay11);
  graph2.makeConnection(lay5, lay11);
  graph.makeConnection(lay6, lay12);
  graph2.makeConnection(lay6, lay12);
  graph.makeConnection(lay6, lay13);
  graph2.makeConnection(lay6, lay13);
  graph.makeConnection(lay7, lay14);
  graph2.makeConnection(lay7, lay14);
  graph.makeConnection(lay7, lay15);
  graph2.makeConnection(lay7, lay15_alt);
  graph.makeConnection(lay8, lay16);
  graph2.makeConnection(lay8, lay16);
  graph.makeConnection(lay8, lay17);
  graph2.makeConnection(lay8, lay17);
  graph.setOutput(lay17, output);
  graph2.setOutput(lay17, output);
  RuntimeOptions opt;
  opt.backend = Backend::kOneDnn;
  ASSERT_NO_THROW(graph.clone(graph_c, output));
  ASSERT_NO_THROW(graph2.clone(graph2_c, output, opt));
}

TEST(graph, check_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 1);
}

TEST(graph, check_connection_remove) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.removeConnection(fcLayer->getID(), ewLayer->getID());
  graph.removeConnection(inputLayer->getID(), fcLayer->getID());

  ASSERT_EQ(graph.areLayerNext(fcLayer, ewLayer), 0);
  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 0);
}

TEST(graph, check_connection_remove_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  ASSERT_ANY_THROW(graph.removeConnection(999, -1));
}

TEST(graph, check_connection_remove_no_edge) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  ASSERT_ANY_THROW(graph.removeConnection(0, 2));
}

TEST(graph, check_connection_double_remove_throw) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.removeConnection(fcLayer->getID(), ewLayer->getID());
  ASSERT_ANY_THROW(graph.removeConnection(fcLayer->getID(), ewLayer->getID()));
}

TEST(graph, check_layer_remove) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.removeSingleLayer(fcLayer->getID());

  ASSERT_EQ(graph.areLayerNext(inputLayer, fcLayer), 0);
  ASSERT_ANY_THROW(graph.areLayerNext(fcLayer, ewLayer));
}

TEST(graph, check_layer_remove_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  ASSERT_ANY_THROW(graph.removeSingleLayer(999));
  ASSERT_ANY_THROW(graph.removeSingleLayer(-1));
}

TEST(graph, check_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, ewLayer);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.setOutput(fcLayer2, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer, fcLayer2), 1);
}

TEST(graph, check_connection_when_not_connection) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto inputLayer = std::make_shared<InputLayer>();
  auto ewLayer = std::make_shared<EWLayer>();
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(inputLayer, input);
  graph.makeConnection(inputLayer, fcLayer);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.setOutput(fcLayer2, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer, ewLayer), false);

  graph.makeConnection(fcLayer, ewLayer);

  ASSERT_EQ(graph.areLayerNext(fcLayer, ewLayer), true);
}

TEST(graph, check_connection_when_not_connection1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer, fcLayer), 0);
}

TEST(graph, check_connection_when_not_connection2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);

  ASSERT_EQ(graph.areLayerNext(fcLayer2, fcLayer4), 0);
}

TEST(graph, set_split_distribution) {
  Graph graph;
  std::vector<std::vector<std::pair<int, int>>> split_dist = {
      {{1, 0}, {2, 1}}, {{3, 0}, {4, 0}, {5, 1}}};
  graph.setSplitDistribution(split_dist);
  SUCCEED();
}

TEST(graph, set_input_null_layer) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  EXPECT_THROW(graph.setInput(nullptr, input), std::invalid_argument);
}

TEST(graph, make_connection_null_layers) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  auto valid_layer = std::make_shared<InputLayer>();

  EXPECT_THROW(graph.makeConnection(nullptr, valid_layer),
               std::invalid_argument);
  EXPECT_THROW(graph.makeConnection(valid_layer, nullptr),
               std::invalid_argument);
  EXPECT_THROW(graph.makeConnection(nullptr, nullptr), std::invalid_argument);
}

TEST(graph, make_connection_same_layer) {
  Graph graph;
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  auto layer = std::make_shared<InputLayer>();
  graph.setInput(layer, input);

  EXPECT_THROW(graph.makeConnection(layer, layer), std::out_of_range);
}

TEST(graph, set_output_null_layer) {
  Graph graph;
  Tensor output;
  EXPECT_THROW(graph.setOutput(nullptr, output), std::invalid_argument);
}

TEST(graph, get_inputs_size_invalid_id) {
  Graph graph;
  EXPECT_THROW(static_cast<void>(graph.getInputsSize(1000)),
               std::invalid_argument);
}

TEST(graph, get_layer_from_id_invalid_id) {
  Graph graph;
  EXPECT_THROW(static_cast<void>(graph.getLayerFromID(1000)),
               std::invalid_argument);
}

TEST(graph, complex_graph_with_split_distribution) {
  std::vector<std::vector<std::pair<int, int>>> split_dist = {{{2, 0}, {3, 1}}};

  Graph graph(10, split_dist);
  Tensor input = make_tensor<float>({1.0F, 2.0F, 3.0F, 4.0F}, {2, 2});
  Tensor output;

  auto input_layer = std::make_shared<InputLayer>();
  auto split_layer = std::make_shared<SplitLayer>(1, 2);
  auto ew_layer1 = std::make_shared<EWLayer>("relu");
  auto ew_layer2 = std::make_shared<EWLayer>("sigmoid");
  auto concat_layer = std::make_shared<ConcatLayer>(0);
  graph.setSplitDistribution(split_dist);

  graph.setInput(input_layer, input);
  graph.makeConnection(input_layer, split_layer);
  graph.makeConnection(split_layer, ew_layer1);
  graph.makeConnection(split_layer, ew_layer2);
  graph.makeConnection(ew_layer1, concat_layer);
  graph.makeConnection(ew_layer2, concat_layer);
  graph.setOutput(concat_layer, output);

  ASSERT_TRUE(graph.areLayerNext(input_layer, split_layer));
  ASSERT_TRUE(graph.areLayerNext(split_layer, ew_layer1));
  ASSERT_TRUE(graph.areLayerNext(split_layer, ew_layer2));
  ASSERT_TRUE(graph.areLayerNext(ew_layer1, concat_layer));
  ASSERT_TRUE(graph.areLayerNext(ew_layer2, concat_layer));
}

TEST(graph, outs_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getOutLayers(5)));
}

TEST(graph, outsizes_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getOutputsSize(999)));
}

TEST(graph, inputs_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getInputsSize(999)));
}

TEST(graph, get_layer_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;

  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getLayerFromID(999)));
}

TEST(graph, get_in_layers_out_of_range) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_ANY_THROW(static_cast<void>(graph.getInLayers(999)));
}

TEST(graph, get_in_layers) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  auto fcLayer = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer3 = std::make_shared<FCLayer>(weights, bias);
  auto fcLayer4 = std::make_shared<FCLayer>(weights, bias);

  graph.setInput(fcLayer, input);
  graph.makeConnection(fcLayer, fcLayer2);
  graph.makeConnection(fcLayer2, fcLayer3);
  graph.makeConnection(fcLayer, fcLayer4);
  graph.setOutput(fcLayer4, output);
  ASSERT_NO_THROW(static_cast<void>(graph.getInLayers(0)));
}

std::vector<std::shared_ptr<FCLayer>> init_fc_layers(size_t size,
                                                     Tensor& weights,
                                                     Tensor& bias) {
  std::vector<std::shared_ptr<FCLayer>> result;
  for (size_t i = 0; i < size; i++) {
    result.push_back(std::make_shared<FCLayer>(weights, bias));
  }
  return result;
}

std::vector<std::shared_ptr<EWLayer>> init_ew_layers(size_t size,
                                                     std::string name) {
  std::vector<std::shared_ptr<EWLayer>> result;
  for (size_t i = 0; i < size; i++) {
    result.push_back(std::make_shared<EWLayer>(name));
  }
  return result;
}

TEST(graph_transformations, check_subgraphs_search) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(6, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.setOutput(fcLayers[3], output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);
  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({1, 2}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search1) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(5, weights, bias);
  auto ewLayer1 = std::make_shared<EWLayer>("relu");
  auto ewLayer2 = std::make_shared<EWLayer>("relu");

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.makeConnection(fcLayers[3], ewLayer1);
  graph.setOutput(ewLayer1, output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], ewLayer2);
  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({3, 4}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(7, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[2], fcLayers[0]);
  graph.makeConnection(fcLayers[2], fcLayers[3]);
  graph.setOutput(fcLayers[3], output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);
  subgraph.makeConnection(fcLayers[5], fcLayers[6]);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({0, 1, 2}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search3) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(7, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[2], fcLayers[0]);
  graph.makeConnection(fcLayers[1], fcLayers[3]);
  graph.setOutput(fcLayers[3], output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);
  subgraph.makeConnection(fcLayers[5], fcLayers[6]);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({2, 0, 1}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search4) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(7, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[2], fcLayers[0]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.setOutput(fcLayers[3], output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);
  subgraph.makeConnection(fcLayers[5], fcLayers[6]);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({1, 2, 0}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_search5) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(7, weights, bias);
  auto ewLayer1 = std::make_shared<EWLayer>("relu");
  auto ewLayer2 = std::make_shared<EWLayer>("relu");

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[3], ewLayer1);
  graph.setOutput(ewLayer1, output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);
  subgraph.addSingleLayer(fcLayers[6]);
  subgraph.makeConnection(fcLayers[6], ewLayer2);

  auto res = find_subgraphs(graph, subgraph);
  auto it = std::find(res.begin(), res.end(), std::vector<int>({1, 3, 2, 4}));
  ASSERT_NE(it, res.end());
}

TEST(graph_transformations, check_subgraphs_big_random) {
  const int num_vertices = 1000;
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;
  Graph graph;
  Graph subgraph;

  std::vector<std::shared_ptr<Layer>> layers;
  for (int i = 0; i < num_vertices / 2; i++) {
    layers.push_back(std::make_shared<FCLayer>(weights, bias));
  }
  for (int i = 0; i < num_vertices / 2; i++) {
    layers.push_back(std::make_shared<EWLayer>("relu"));
  }
  layers.push_back(std::make_shared<FCLayer>(weights, bias));
  layers.push_back(std::make_shared<EWLayer>("relu"));
  layers.push_back(std::make_shared<FCLayer>(weights, bias));

  graph.setInput(layers[0], input);
  for (int i = 0; i < num_vertices; i++) {
    int rFirst = rand() % (num_vertices - 1);
    int rSecond = 1 + rand() % (num_vertices - 1);
    if ((rFirst == rSecond) ||
        ((layers[rFirst]->getID() == layers[rSecond]->getID()) &&
         (layers[rFirst]->getID() != 0))) {
      continue;
    }
    if ((layers[rFirst]->getID() >= graph.getLayersCount()) ||
        (rFirst != 0 && layers[rFirst]->getID() == 0)) {
      graph.addSingleLayer(layers[rFirst]);
    }
    graph.makeConnection(layers[rFirst], layers[rSecond]);
  }
  graph.setOutput(layers[num_vertices - 1], output);

  subgraph.setInput(layers[num_vertices], input);
  subgraph.makeConnection(layers[num_vertices], layers[num_vertices + 1]);
  subgraph.makeConnection(layers[num_vertices + 1], layers[num_vertices + 2]);

  std::vector<std::vector<int>> res1 = find_subgraphs(graph, subgraph);
  double res1_time =
      elapsed_time_avg<double, std::milli>(10, find_subgraphs, graph, subgraph);
  std::cerr << "Find subgraphs time in ms " << res1_time << std::endl;
}

class SubgraphTestsParameterized
    : public ::testing::TestWithParam<std::vector<std::tuple<int, int>>> {};

TEST_P(SubgraphTestsParameterized, check_subgraphs_big_random_lines) {
  auto data = GetParam();
  for (size_t m = 0; m < data.size(); m++) {
    std::cerr << "(" << std::get<1>(data[m]) << ") ";
    int num_vertices = std::get<0>(data[m]);
    int num_vertices_sub = std::get<1>(data[m]);
    const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
    Tensor weights = make_tensor<float>(vec1, {3, 2});
    Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
    Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
    Tensor output;
    Graph graph;
    Graph subgraph;
    std::vector<std::shared_ptr<Layer>> layers;
    for (int i = 0; i < num_vertices; i++) {
      layers.push_back(std::make_shared<FCLayer>(weights, bias));
    }
    graph.setInput(layers[0], input);
    for (int i = 0; i < num_vertices - 1; i++) {
      graph.makeConnection(layers[i], layers[i + 1]);
    }
    graph.setOutput(layers[num_vertices - 1], output);

    std::vector<std::shared_ptr<Layer>> temp_layers;
    for (int i = 0; i < num_vertices_sub + 2; i++) {
      temp_layers.push_back(std::make_shared<FCLayer>(weights, bias));
    }
    subgraph.setInput(temp_layers[0], input);
    for (int i = 0; i < num_vertices_sub; i++) {
      subgraph.makeConnection(temp_layers[i], temp_layers[i + 1]);
    }

    double res1_time = elapsed_time_avg<double, std::milli>(1, find_subgraphs,
                                                            graph, subgraph);
    std::cerr << "Find subgraphs time in ms "
              << res1_time / (100 * num_vertices_sub * num_vertices_sub)
              << std::endl;
  }
}

std::vector<std::tuple<int, int>> genVector() {
  std::vector<std::tuple<int, int>> results(10);
  for (size_t i = 0; i < results.size(); i++) {
    results[i] = std::tuple<int, int>(105, 2 + 2 * static_cast<int>(i));
  }
  return results;
}

INSTANTIATE_TEST_SUITE_P(graph_transformations, SubgraphTestsParameterized,
                         ::testing::Values(genVector()));

TEST(graph_transformations, check_subgraphs_replace) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(9, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.setOutput(fcLayers[3], output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);

  res_graph.setInput(fcLayers[6], input);
  res_graph.makeConnection(fcLayers[6], fcLayers[7]);
  std::shared_ptr<Layer> lay = std::make_shared<EWLayer>("relu");
  res_graph.addSingleLayer(lay);
  res_graph.makeConnection(lay, fcLayers[8]);

  Graph res;
  std::shared_ptr<Layer> lay_to = std::make_shared<EWLayer>("relu");
  changed_subgraphs(graph, subgraph, lay_to, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(8, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.addSingleLayer(fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.makeConnection(fcLayers[3], fcLayers[4]);
  graph.setOutput(fcLayers[4], output);

  subgraph.setInput(fcLayers[5], input);
  subgraph.makeConnection(fcLayers[5], fcLayers[6]);

  std::shared_ptr<Layer> lay = std::make_shared<EWLayer>("relu");
  std::shared_ptr<Layer> lay2 = std::make_shared<EWLayer>("relu");
  res_graph.setInput(lay2, input);
  res_graph.addSingleLayer(lay);
  res_graph.makeConnection(lay, fcLayers[7]);

  Graph res;
  std::shared_ptr<Layer> lay_to = std::make_shared<EWLayer>("relu");
  changed_subgraphs(graph, subgraph, lay_to, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace3) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(10, weights, bias);
  Shape shapepool({2, 2});
  std::shared_ptr<Layer> pool1 =
      std::make_shared<PoolingLayer>(shapepool, "max");
  std::shared_ptr<Layer> pool2 =
      std::make_shared<PoolingLayer>(shapepool, "max");
  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[2], fcLayers[3]);
  graph.makeConnection(fcLayers[3], fcLayers[4]);
  graph.makeConnection(fcLayers[4], pool1);
  graph.makeConnection(fcLayers[2], pool1);
  graph.setOutput(pool1, output);

  subgraph.setInput(fcLayers[5], input);
  subgraph.makeConnection(fcLayers[5], fcLayers[7]);
  subgraph.makeConnection(fcLayers[7], pool2);
  subgraph.addSingleLayer(fcLayers[6]);
  subgraph.makeConnection(fcLayers[6], fcLayers[5]);
  subgraph.makeConnection(fcLayers[6], pool2);

  std::shared_ptr<Layer> lay = std::make_shared<EWLayer>("relu");
  res_graph.setInput(fcLayers[8], input);
  res_graph.makeConnection(fcLayers[8], fcLayers[9]);
  res_graph.makeConnection(fcLayers[9], lay);

  Graph res;
  std::shared_ptr<Layer> lay_to = std::make_shared<EWLayer>("relu");
  changed_subgraphs(graph, subgraph, lay_to, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace4) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(8, weights, bias);
  std::shared_ptr<Layer> ewLayer1 = std::make_shared<EWLayer>("relu");

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[2], fcLayers[3]);
  graph.makeConnection(fcLayers[3], ewLayer1);
  graph.makeConnection(ewLayer1, fcLayers[4]);
  graph.setOutput(fcLayers[4], output);

  subgraph.setInput(fcLayers[5], input);
  subgraph.makeConnection(fcLayers[5], fcLayers[6]);

  std::shared_ptr<Layer> lay = std::make_shared<EWLayer>("relu");
  std::shared_ptr<Layer> lay2 = std::make_shared<EWLayer>("relu");
  std::shared_ptr<Layer> ewLayer2 = std::make_shared<EWLayer>("relu");

  res_graph.setInput(lay, input);
  res_graph.makeConnection(lay, lay2);
  res_graph.makeConnection(lay2, ewLayer2);
  res_graph.makeConnection(ewLayer2, fcLayers[7]);

  Graph res;
  std::shared_ptr<Layer> lay_to = std::make_shared<EWLayer>("relu");
  changed_subgraphs(graph, subgraph, lay_to, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace5) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  std::shared_ptr<Layer> fcLayer1 = std::make_shared<FCLayer>(weights, bias);
  std::shared_ptr<Layer> fcLayer2 = std::make_shared<FCLayer>(weights, bias);
  std::vector<std::shared_ptr<EWLayer>> ewLayers = init_ew_layers(14, "relu");

  graph.setInput(fcLayer1, input);
  for (int i = 0; i < 4; i++) {
    graph.makeConnection(fcLayer1, ewLayers[i]);
  }
  for (int i = 0; i < 4; i++) {
    graph.makeConnection(ewLayers[i], ewLayers[i + 4]);
  }
  graph.setOutput(ewLayers[7], output);

  subgraph.setInput(ewLayers[8], input);
  subgraph.makeConnection(ewLayers[8], ewLayers[9]);

  res_graph.setInput(fcLayer2, input);
  for (int i = 0; i < 4; i++) {
    res_graph.makeConnection(fcLayer2, ewLayers[10 + i]);
  }

  Graph res;
  std::shared_ptr<Layer> lay_to = std::make_shared<EWLayer>("relu");
  changed_subgraphs(graph, subgraph, lay_to, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace_s) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  Graph subgraph2;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(12, weights, bias);

  std::vector<std::shared_ptr<FCLayer>> fcLayers_to =
      init_fc_layers(3, weights, bias);
  Shape shape = {2, 2};

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.setOutput(fcLayers[3], output);

  subgraph.setInput(fcLayers[4], input);
  subgraph.makeConnection(fcLayers[4], fcLayers[5]);

  subgraph2.setInput(fcLayers_to[0], input);
  subgraph2.makeConnection(fcLayers_to[0], fcLayers_to[1]);
  std::shared_ptr<Layer> pool_to = std::make_shared<PoolingLayer>(shape, "max");
  subgraph2.makeConnection(fcLayers_to[1], pool_to);
  subgraph2.makeConnection(pool_to, fcLayers_to[2]);

  res_graph.setInput(fcLayers[6], input);
  res_graph.makeConnection(fcLayers[6], fcLayers[7]);
  std::shared_ptr<Layer> pool = std::make_shared<PoolingLayer>(shape, "max");
  res_graph.addSingleLayer(fcLayers[8]);
  res_graph.makeConnection(fcLayers[8], fcLayers[9]);
  res_graph.makeConnection(fcLayers[9], pool);
  res_graph.makeConnection(pool, fcLayers[10]);
  res_graph.makeConnection(fcLayers[10], fcLayers[11]);

  Graph res;
  changed_subgraphs(graph, subgraph, subgraph2, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace_s2) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  Graph subgraph2;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(21, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.addSingleLayer(fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[0], fcLayers[3]);
  graph.makeConnection(fcLayers[3], fcLayers[4]);
  graph.setOutput(fcLayers[4], output);

  subgraph.setInput(fcLayers[5], input);
  subgraph.makeConnection(fcLayers[5], fcLayers[6]);

  subgraph2.setInput(fcLayers[8], input);
  subgraph2.makeConnection(fcLayers[8], fcLayers[9]);
  subgraph2.makeConnection(fcLayers[8], fcLayers[10]);
  subgraph2.makeConnection(fcLayers[9], fcLayers[11]);
  subgraph2.makeConnection(fcLayers[10], fcLayers[11]);

  res_graph.setInput(fcLayers[12], input);
  res_graph.addSingleLayer(fcLayers[13]);
  res_graph.makeConnection(fcLayers[12], fcLayers[14]);
  res_graph.makeConnection(fcLayers[12], fcLayers[15]);
  res_graph.makeConnection(fcLayers[14], fcLayers[16]);
  res_graph.makeConnection(fcLayers[15], fcLayers[16]);
  res_graph.makeConnection(fcLayers[16], fcLayers[17]);
  res_graph.makeConnection(fcLayers[13], fcLayers[18]);
  res_graph.makeConnection(fcLayers[13], fcLayers[19]);
  res_graph.makeConnection(fcLayers[18], fcLayers[20]);
  res_graph.makeConnection(fcLayers[19], fcLayers[20]);

  Graph res;
  changed_subgraphs(graph, subgraph, subgraph2, res, output);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace_s3) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  Graph subgraph2;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(26, weights, bias);
  std::vector<std::shared_ptr<EWLayer>> ewLayers = init_ew_layers(10, "relu");
  graph.setInput(ewLayers[6], input);
  graph.addSingleLayer(fcLayers[1]);
  graph.makeConnection(ewLayers[6], ewLayers[0]);
  graph.makeConnection(ewLayers[6], fcLayers[3]);
  graph.makeConnection(fcLayers[1], ewLayers[0]);
  graph.makeConnection(fcLayers[3], fcLayers[4]);
  graph.makeConnection(ewLayers[0], ewLayers[1]);
  graph.makeConnection(ewLayers[1], fcLayers[6]);
  graph.makeConnection(ewLayers[1], fcLayers[7]);
  graph.makeConnection(fcLayers[4], fcLayers[7]);
  graph.setOutput(fcLayers[7], output);

  subgraph.setInput(fcLayers[8], input);
  subgraph.makeConnection(fcLayers[8], fcLayers[9]);
  subgraph.addSingleLayer(ewLayers[4]);
  subgraph.makeConnection(ewLayers[4], ewLayers[5]);

  subgraph2.setInput(ewLayers[8], input);
  subgraph2.addSingleLayer(fcLayers[13]);
  subgraph2.makeConnection(ewLayers[8], fcLayers[14]);
  subgraph2.makeConnection(fcLayers[13], fcLayers[14]);
  subgraph2.makeConnection(fcLayers[14], ewLayers[9]);
  subgraph2.makeConnection(fcLayers[14], fcLayers[16]);

  res_graph.setInput(fcLayers[17], input);
  res_graph.addSingleLayer(ewLayers[7]);
  res_graph.makeConnection(fcLayers[17], ewLayers[2]);
  res_graph.makeConnection(ewLayers[7], ewLayers[2]);
  res_graph.makeConnection(ewLayers[7], fcLayers[20]);
  res_graph.makeConnection(ewLayers[2], fcLayers[21]);
  res_graph.makeConnection(fcLayers[20], fcLayers[21]);
  res_graph.makeConnection(fcLayers[21], fcLayers[22]);
  res_graph.makeConnection(fcLayers[21], ewLayers[3]);
  res_graph.makeConnection(ewLayers[3], fcLayers[24]);
  res_graph.makeConnection(fcLayers[22], fcLayers[25]);
  res_graph.makeConnection(ewLayers[3], fcLayers[25]);
  IOOrder order;
  order.in_order = std::vector<int>({1, 0});
  order.out_order = std::vector<int>({1, 0});
  Graph res;
  changed_subgraphs(graph, subgraph, subgraph2, res, output, RuntimeOptions(),
                    order);
  ASSERT_FALSE(find_subgraphs(res, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace_s4) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  Graph subgraph2;
  Graph subgraph_to;
  Graph subgraph_to2;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(28, weights, bias);

  graph.setInput(fcLayers[0], input);
  graph.makeConnection(fcLayers[0], fcLayers[1]);
  graph.makeConnection(fcLayers[1], fcLayers[2]);
  graph.makeConnection(fcLayers[2], fcLayers[3]);
  graph.makeConnection(fcLayers[3], fcLayers[4]);
  graph.addSingleLayer(fcLayers[5]);
  graph.makeConnection(fcLayers[5], fcLayers[4]);
  graph.makeConnection(fcLayers[0], fcLayers[5]);
  graph.setOutput(fcLayers[5], output);

  subgraph.setInput(fcLayers[6], input);
  subgraph.makeConnection(fcLayers[6], fcLayers[7]);
  subgraph.makeConnection(fcLayers[6], fcLayers[8]);

  subgraph2.setInput(fcLayers[9], input);
  subgraph2.addSingleLayer(fcLayers[10]);
  subgraph2.makeConnection(fcLayers[9], fcLayers[11]);
  subgraph2.makeConnection(fcLayers[10], fcLayers[11]);

  subgraph_to.setInput(fcLayers[12], input);
  subgraph_to.makeConnection(fcLayers[12], fcLayers[13]);
  subgraph_to.makeConnection(fcLayers[13], fcLayers[14]);
  subgraph_to.makeConnection(fcLayers[13], fcLayers[15]);

  subgraph_to2.setInput(fcLayers[16], input);
  subgraph_to2.addSingleLayer(fcLayers[17]);
  subgraph_to2.makeConnection(fcLayers[16], fcLayers[18]);
  subgraph_to2.makeConnection(fcLayers[17], fcLayers[18]);
  subgraph_to2.makeConnection(fcLayers[18], fcLayers[19]);

  res_graph.setInput(fcLayers[20], input);
  res_graph.makeConnection(fcLayers[20], fcLayers[21]);
  res_graph.makeConnection(fcLayers[21], fcLayers[22]);
  res_graph.makeConnection(fcLayers[21], fcLayers[23]);
  res_graph.makeConnection(fcLayers[22], fcLayers[24]);
  res_graph.makeConnection(fcLayers[24], fcLayers[25]);
  res_graph.makeConnection(fcLayers[25], fcLayers[26]);
  res_graph.makeConnection(fcLayers[23], fcLayers[26]);
  res_graph.makeConnection(fcLayers[26], fcLayers[27]);
  Graph graph2;
  graph.clone(graph2, output);
  Graph res1;
  changed_subgraphs(graph, subgraph, subgraph_to, res1, output);
  Graph res2;
  changed_subgraphs(res1, subgraph2, subgraph_to2, res2, output);

  Graph res2_1;
  changed_subgraphs(graph2, subgraph2, subgraph_to2, res2_1, output);
  Graph res1_1;
  changed_subgraphs(res2_1, subgraph, subgraph_to, res1_1, output);

  ASSERT_FALSE(find_subgraphs(res2, res_graph).empty());
  ASSERT_FALSE(find_subgraphs(res1_1, res_graph).empty());
}

TEST(graph_transformations, check_subgraphs_replace_s5) {
  const std::vector<float> vec1 = {2.0F, 1.5F, 0.1F, 1.9F, 0.0F, 5.5F};
  Tensor weights = make_tensor<float>(vec1, {3, 2});
  Tensor bias = make_tensor<float>({0.5F, 0.5F, 1.0F});
  Tensor input = make_tensor<float>({1.0F, 2.0F}, {2});
  Tensor output;

  Graph graph;
  Graph res_graph;
  Graph subgraph;
  Graph subgraph2;
  Graph subgraph_to;
  std::vector<std::shared_ptr<FCLayer>> fcLayers =
      init_fc_layers(3, weights, bias);

  std::vector<std::shared_ptr<EWLayer>> ewLayers = init_ew_layers(16, "relu");

  graph.setInput(fcLayers[0], input);
  for (int i = 0; i < 4; i++) {
    graph.makeConnection(fcLayers[0], ewLayers[i]);
  }
  for (int i = 0; i < 4; i++) {
    graph.makeConnection(ewLayers[i], ewLayers[i + 4]);
  }
  graph.setOutput(ewLayers[7], output);

  subgraph.setInput(ewLayers[8], input);
  subgraph.makeConnection(ewLayers[8], ewLayers[9]);

  subgraph2.setInput(fcLayers[1], input);
  subgraph2.makeConnection(fcLayers[1], ewLayers[10]);

  subgraph_to.setInput(ewLayers[11], input);

  res_graph.setInput(fcLayers[2], input);
  res_graph.makeConnection(fcLayers[2], ewLayers[12]);
  res_graph.makeConnection(fcLayers[2], ewLayers[13]);
  res_graph.makeConnection(fcLayers[2], ewLayers[14]);
  res_graph.addSingleLayer(ewLayers[15]);

  Graph res;
  std::shared_ptr<Layer> lay_to = std::make_shared<EWLayer>("relu");
  changed_subgraphs(graph, subgraph, lay_to, res, output);
  Graph res2;
  std::shared_ptr<Layer> lay_to2 = std::make_shared<EWLayer>("relu");
  changed_subgraphs(res, subgraph2, lay_to2, res2, output);
  ASSERT_FALSE(find_subgraphs(res2, res_graph).empty());
}
