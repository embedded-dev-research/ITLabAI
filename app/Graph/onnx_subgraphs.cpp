#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <unordered_map>

#include "build.hpp"
#include "graph_transformations/graph_transformations.hpp"
#include "layers_fused/ConvRelu.hpp"
#include "layers_fused/DenseNetPath.hpp"
#include "perf/benchmarking.hpp"

using namespace it_lab_ai;

void print_split_dist(const Graph& graph) {
  auto split_dist = graph.getSplitDistribution();
  for (int i = 0; i < split_dist.size(); i++) {
    std::cout << "Split #" << i + 1 << ": ";
    for (int j = 0; j < split_dist[i].size(); j++) {
      std::cout << "(" << split_dist[i][j].first << ", "
                << split_dist[i][j].second << ") ";
    }
    std::cout << std::endl;
  }
}

void alexnet_inf_careless(Graph& graph, const RuntimeOptions& options,
                          Tensor& input, Tensor& output) {
  auto* o = new Tensor(output);
  auto* i = new Tensor(input);
  graph.setOutput(*o);
  graph.setInput(graph.getLayerFromID(0), *i);
  graph.inference(options);
  if (graph.getLayersCount() == 0) {
    throw std::runtime_error("No layers");
  }
  //std::cout << *o;
}

void create_def_graph(Graph& graph) {
  std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                891, 957,  1027, 973,  1008};
  size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
  int count_pic = static_cast<int>(sum) + 10;
  std::vector<float> res(count_pic * 28 * 28, 1.0F);
  Tensor input;
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor output = make_tensor(vec, sh1);

  Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
  Tensor t = make_tensor<float>(res, sh);
  input = t;

  RuntimeOptions options;
  build_graph_linear(graph, input, output, options, true, false);
}

void create_def_graph_gn(Graph& graph) {
  Tensor input = prepare_image(
      cv::imread(IMAGENET_PATH + std::string("1.png")),
      get_input_shape_from_json(MODEL_PATH_GOOGLENET_ONNX), "google");
  Tensor output = make_tensor(std::vector<float>({0.0F}));
  RuntimeOptions options;
  build_graph(graph, input, output, MODEL_PATH_GOOGLENET_ONNX, options, true);
}

void create_def_graph_yolo(Graph& graph) {
  Tensor input = prepare_image(
      cv::imread(IMAGENET_PATH + std::string("1.png")),
      get_input_shape_from_json(MODEL_PATH_YOLO11NET_ONNX), "yolo");
  Tensor output = make_tensor(std::vector<float>({0.0F}));
  RuntimeOptions options;
  build_graph(graph, input, output, MODEL_PATH_YOLO11NET_ONNX, options, true);
}

void create_changed_graph(Graph& graph2) {
  std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                891, 957,  1027, 973,  1008};
  size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
  int count_pic = static_cast<int>(sum) + 10;
  std::vector<float> res(count_pic * 28 * 28, 1.0F);
  Tensor input;
  Shape sh1({1, 5, 5, 3});
  std::vector<float> vec;
  vec.reserve(75);
  for (int i = 0; i < 75; ++i) {
    vec.push_back(3);
  }
  Tensor output = make_tensor(vec, sh1);

  Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
  Tensor t = make_tensor<float>(res, sh);
  input = t;

  RuntimeOptions options;
  Graph graph;
  graph2.clone(graph, output, options);
  Graph subgraph;
  std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
  std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
  subgraph.setInput(layer_0, input);
  subgraph.makeConnection(layer_0, layer_1);
  std::shared_ptr<Layer> layer_to = std::make_shared<ConvReluLayer>(
      std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
  changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
}

void create_changed_graph_gn(Graph& graph2) {
  Tensor input = prepare_image(
      cv::imread(IMAGENET_PATH + std::string("1.png")),
      get_input_shape_from_json(MODEL_PATH_GOOGLENET_ONNX), "google");
  Tensor output = make_tensor(std::vector<float>({0.0F}));
  RuntimeOptions options;
  Graph graph;
  graph2.clone(graph, output, options);
  Graph subgraph;
  std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
  std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
  subgraph.setInput(layer_0, input);
  subgraph.makeConnection(layer_0, layer_1);
  std::shared_ptr<Layer> layer_to = std::make_shared<ConvReluLayer>(
      std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
  changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
}

void create_changed_graph_yolo(Graph& graph2) {
  Tensor input = prepare_image(
      cv::imread(IMAGENET_PATH + std::string("1.png")),
      get_input_shape_from_json(MODEL_PATH_YOLO11NET_ONNX), "yolo");
  Tensor output = make_tensor(std::vector<float>({0.0F}));
  RuntimeOptions options;
  Graph graph;
  graph2.clone(graph, output, options);
  Graph subgraph;
  std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
  std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("sigmoid");
  std::shared_ptr<Layer> layer_2 =
      std::make_shared<BinaryOpLayer>(BinaryOpLayer::Operation::kMul);
  subgraph.setInput(layer_0, input);
  subgraph.makeConnection(layer_0, layer_1);
  subgraph.makeConnection(layer_1, layer_2);
  subgraph.makeConnection(layer_0, layer_2);
  std::shared_ptr<Layer> layer_to = std::make_shared<ConvSigmMulLayer>(
      std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
  changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
}

void alexnet_comparison(int type = 3, int whoonly = 3) {
  if (type == 0) {
    Tensor input =
        prepare_image(cv::imread(IMAGENET_PATH + std::string("1.png")),
                      get_input_shape_from_json(MODEL_PATH_DENSENET_ONNX));
    Tensor output = make_tensor(std::vector<float>({0.0F}));
    Tensor input_c = input;
    Tensor output_c = make_tensor(std::vector<float>({0.0F}));
    RuntimeOptions options;
    Graph graph;
    Graph graph2;
    build_graph(graph, input, output, MODEL_PATH_DENSENET_ONNX, options, false);
    Graph subgraph;
    Tensor scale = make_tensor(std::vector<float>({1.0F}));
    std::shared_ptr<Layer> layer_0 =
        std::make_shared<BatchNormalizationLayer>(scale, scale, scale, scale);
    std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
    std::shared_ptr<Layer> layer_2 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_3 = std::make_shared<EWLayer>("relu");
    std::shared_ptr<Layer> layer_4 = std::make_shared<ConvolutionalLayer>();
    subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    subgraph.makeConnection(layer_1, layer_2);
    subgraph.makeConnection(layer_2, layer_3);
    subgraph.makeConnection(layer_3, layer_4);
    std::shared_ptr<Layer> layer_to = std::make_shared<DenseNetPath>();
    changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
    double time1 = 0.0;
    double time2 = 0.0;
    if (whoonly == 1 || whoonly == 3) {
      time1 = elapsed_time_avg<double, std::milli>(
          10, alexnet_inf_careless, graph, options, input_c, output_c);
      print_time_stats(graph);
    }
    if (whoonly == 2 || whoonly == 3) {
      time2 = elapsed_time_avg<double, std::milli>(
          10, alexnet_inf_careless, graph2, options, input_c, output_c);
      print_time_stats(graph2);
    }
    std::cout << time1 << " for unchanged graph\n";
    std::cout << time2 << " for convrelu graph\n";
  } else if (type == 2) {
    Tensor input = prepare_image(
        cv::imread(IMAGENET_PATH + std::string("1.png")),
        get_input_shape_from_json(MODEL_PATH_GOOGLENET_ONNX), "google");
    Tensor output = make_tensor(std::vector<float>({0.0F}));
    Tensor input_c = input;
    Tensor output_c = make_tensor(std::vector<float>({0.0F}));
    RuntimeOptions options;
    Graph graph;
    Graph graph2;
    build_graph(graph, input, output, MODEL_PATH_GOOGLENET_ONNX, options, true);
    Graph subgraph;
    std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
    subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    std::shared_ptr<Layer> layer_to = std::make_shared<ConvReluLayer>(
        std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
    changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
    double time1 = 0.0;
    double time2 = 0.0;
    if (whoonly == 1 || whoonly == 3) {
      time1 = elapsed_time_avg<double, std::milli>(
          10, alexnet_inf_careless, graph, options, input_c, output_c);
      print_time_stats(graph);
    }
    if (whoonly == 2 || whoonly == 3) {
      time2 = elapsed_time_avg<double, std::milli>(
          10, alexnet_inf_careless, graph2, options, input_c, output_c);
      print_time_stats(graph2);
    }
    std::cout << time1 << " for unchanged graph\n";
    std::cout << time2 << " for convrelu graph\n";
  } else if (type == 3) {
    //std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
    //                              891, 957,  1027, 973,  1008};
    std::vector<size_t> counts = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
    size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
    int count_pic = static_cast<int>(sum) + 10;
    std::vector<float> res(count_pic * 28 * 28, 1.0F);
    Tensor input;
    Shape sh1({1, 5, 5, 3});
    std::vector<float> vec;
    vec.reserve(75);
    for (int i = 0; i < 75; ++i) {
      vec.push_back(3);
    }
    Tensor output = make_tensor(vec, sh1);

    Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
    Tensor t = make_tensor<float>(res, sh);
    input = t;

    RuntimeOptions options;
    Graph graph;
    Graph graph2;
    build_graph_linear(graph, input, output, options, true, false);
    Graph subgraph;
    std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
    subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    std::shared_ptr<Layer> layer_to = std::make_shared<ConvReluLayer>(
        std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
    changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
    Tensor input_c = input;
    Tensor output_c = output;
    double time1 = 0.0;
    double time2 = 0.0;
    if (whoonly == 1 || whoonly == 3) {
      time1 = elapsed_time_avg<double, std::milli>(
          10, alexnet_inf_careless, graph, options, input_c, output_c);
      print_time_stats(graph);
    }
    if (whoonly == 2 || whoonly == 3) {
      time2 = elapsed_time_avg<double, std::milli>(
          10, alexnet_inf_careless, graph2, options, input_c, output_c);
      print_time_stats(graph2);
    }
    std::cout << time1 << " for unchanged graph\n";
    std::cout << time2 << " for convrelu graph\n";
  } else if (type == 4) {
    Tensor input = prepare_image(
        cv::imread(IMAGENET_PATH + std::string("1.png")),
        get_input_shape_from_json(MODEL_PATH_YOLO11NET_ONNX), "yolo");
    Tensor output = make_tensor(std::vector<float>({0.0F}));
    Tensor input_c = input;
    Tensor output_c = make_tensor(std::vector<float>({0.0F}));
    RuntimeOptions options;
    Graph graph;
    Graph graph2;
    build_graph(graph, input, output, MODEL_PATH_YOLO11NET_ONNX, options, true);
    Graph subgraph;
    std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
    std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("sigmoid");
    //std::shared_ptr<Layer> layer_2 =
    //    std::make_shared<BinaryOpLayer>(BinaryOpLayer::Operation::kMul);
    //subgraph.setInput(layer_0, input);
    subgraph.makeConnection(layer_0, layer_1);
    //subgraph.makeConnection(layer_1, layer_2);
    //subgraph.makeConnection(layer_0, layer_2);
    std::shared_ptr<Layer> layer_to = std::make_shared<ConvReluLayer>(
        std::dynamic_pointer_cast<ConvolutionalLayer>(layer_0));
    changed_subgraphs(graph, subgraph, layer_to, graph2, input, options);
    print_split_dist(graph);
    print_split_dist(graph2);
    double time1 = 0.0;
    double time2 = 0.0;
    try {
      if (whoonly == 1 || whoonly == 3) {
        time1 = elapsed_time_avg<double, std::milli>(
            5, alexnet_inf_careless, graph, options, input_c, output_c);
        print_time_stats(graph);
      }
      if (whoonly == 2 || whoonly == 3) {
        time2 = elapsed_time_avg<double, std::milli>(
            5, alexnet_inf_careless, graph2, options, input_c, output_c);
        print_time_stats(graph2);
      }
    } catch (std::exception& e) {
      std::cout << e.what();
    }
    std::cout << time1 << " for unchanged graph\n";
    std::cout << time2 << " for convsigmmul graph\n";
  }
}

int main(int argc, char* argv[]) {
  //int type = (argc > 1) ? (int)(argv[1][0]-'0') : 0;
  if (argc > 1) {
    argv;
  }
  int type = 1;
  int type2 = 2;
  int whoonly = 3;
  std::cout << "Type of network (1 - densenet, 2 - resnet, 3 - googlenet, 4 - alexnet, 5 - yolo): ";
  std::cin >> type;
  type--;
  std::cout << "Type of analyze (1 - unchanged graph only, 2 - changed graph "
               "only, 3 - both): ";
  std::cin >> whoonly;
  std::cout << "(1) Test subgraph search algorithm\n";
  if (type != 1) {
    std::cout
        << "(2) Compare inference time for unchanged and changed subgraphs\n";
  }
  if (type >= 2) {
    std::cout
        << "(3) Compare memory usage for unchanged and changed subgraphs\n";
  }
  std::cin >> type2;
  Tensor input = make_tensor(std::vector<float>({0.0F}));
  RuntimeOptions options;
  if (type2 == 2) {
    alexnet_comparison(type, whoonly);
  }
  if (type2 == 3 && type >= 2) {
    Graph graph;
    if (type == 2 || type == 4) {
      create_def_graph_gn(graph);
    } else if (type == 3) {
      create_def_graph(graph);
    } /*else if (type == 4) {
      create_def_graph_yolo(graph);
    }*/
    std::cout << "End of graph creation: def\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    if (type == 2) {
      create_changed_graph_gn(graph);
    } else if (type == 3) {
      create_changed_graph(graph);
    } else if (type == 4) {
      create_changed_graph_yolo(graph);
    }
    std::cout << "End of graph creation: changed\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  }
  if (type2 == 1) {
    if (type == 0) {
      Graph graph1;
      build_graph(graph1, input, input, MODEL_PATH_DENSENET_ONNX, options,
                  false);

      Graph subgraph;
      Tensor scale = make_tensor(std::vector<float>({1.0F}));
      std::shared_ptr<Layer> layer_0 =
          std::make_shared<BatchNormalizationLayer>(scale, scale, scale, scale);
      std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
      std::shared_ptr<Layer> layer_2 = std::make_shared<ConvolutionalLayer>();
      std::shared_ptr<Layer> layer_3 = std::make_shared<EWLayer>("relu");
      std::shared_ptr<Layer> layer_4 = std::make_shared<ConvolutionalLayer>();
      subgraph.setInput(layer_0, input);
      subgraph.makeConnection(layer_0, layer_1);
      subgraph.makeConnection(layer_1, layer_2);
      subgraph.makeConnection(layer_2, layer_3);
      subgraph.makeConnection(layer_3, layer_4);

      Graph subgraph2;
      std::shared_ptr<Layer> layer_5 = std::make_shared<ConcatLayer>();
      std::shared_ptr<Layer> layer_6 =
          std::make_shared<PoolingLayer>(Shape({1, 1, 1}), "max");
      std::shared_ptr<Layer> layer_7 = std::make_shared<ConvolutionalLayer>();
      subgraph2.setInput(layer_6, input);
      subgraph2.makeConnection(layer_6, layer_5);
      subgraph2.addSingleLayer(layer_7);
      subgraph2.makeConnection(layer_7, layer_5);

      auto vec = find_subgraphs(graph1, subgraph);
      auto vec2 = find_subgraphs(graph1, subgraph2);

      auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                       graph1, subgraph);
      auto time2 = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                        graph1, subgraph2);

      for (int i = 0; i < vec.size(); i++) {
        std::cout << i + 1 << ") ";
        for (int j : vec[i]) {
          std::cout << j << ' ';
        }
        std::cout << '\n';
      }
      std::cout << "Time for DenseNet BN -> ReLU -> Conv -> ReLU -> Conv: " << time << '\n';

      for (int i = 0; i < vec2.size(); i++) {
        std::cout << i + 1 << ") ";
        for (int j : vec2[i]) {
          std::cout << j << ' ';
        }
        std::cout << '\n';
      }
      std::cout << "Time for DenseNet Concat with Pool and Conv: " << time2 << '\n';
    } else if (type == 1) {
      Graph graph1;
      build_graph(graph1, input, input, MODEL_PATH_RESNET_ONNX, options, false);

      Graph subgraph;
      std::shared_ptr<Layer> layer_0 = std::make_shared<TransposeLayer>();
      std::shared_ptr<Layer> layer_1 = std::make_shared<SoftmaxLayer>();
      std::shared_ptr<Layer> layer_2 = std::make_shared<ReshapeLayer>();
      std::shared_ptr<Layer> layer_3 = std::make_shared<ReshapeLayer>();
      std::shared_ptr<Layer> layer_4 = std::make_shared<ReshapeLayer>();
      subgraph.setInput(layer_0, input);
      subgraph.makeConnection(layer_0, layer_1);
      subgraph.makeConnection(layer_1, layer_2);
      subgraph.makeConnection(layer_2, layer_3);
      subgraph.makeConnection(layer_3, layer_4);

      auto vec = find_subgraphs(graph1, subgraph);

      auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                       graph1, subgraph);
      for (int i = 0; i < vec.size(); i++) {
        std::cout << i + 1 << ") ";
        for (int j : vec[i]) {
          std::cout << j << ' ';
        }
        std::cout << '\n';
      }
      std::cout << "Time for ResNet Transpose -> SoftMax -> Reshape -> Reshape -> Reshape: " << time << '\n';
    } else if (type == 2) {
      Graph graph1;
      build_graph(graph1, input, input, MODEL_PATH_GOOGLENET_ONNX, options,
                  false);

      Graph subgraph;
      Shape shape(2);
      std::shared_ptr<Layer> layer_0 = std::make_shared<ConcatLayer>();
      std::shared_ptr<Layer> layer_1 = std::make_shared<ConvolutionalLayer>();
      std::shared_ptr<Layer> layer_2 = std::make_shared<EWLayer>("relu");
      std::shared_ptr<Layer> layer_3 = std::make_shared<EWLayer>("relu");
      std::shared_ptr<Layer> layer_4 = std::make_shared<ConvolutionalLayer>();
      std::shared_ptr<Layer> layer_5 = std::make_shared<ConvolutionalLayer>();
      std::shared_ptr<Layer> layer_6 =
          std::make_shared<PoolingLayer>(shape, "max");
      subgraph.setInput(layer_0, input);
      subgraph.makeConnection(layer_0, layer_1);
      subgraph.makeConnection(layer_0, layer_4);
      subgraph.makeConnection(layer_0, layer_5);
      subgraph.makeConnection(layer_0, layer_6);
      subgraph.makeConnection(layer_4, layer_2);
      subgraph.makeConnection(layer_5, layer_3);

      auto vec = find_subgraphs(graph1, subgraph);

      auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                       graph1, subgraph);
      for (int i = 0; i < vec.size(); i++) {
        std::cout << i + 1 << ") ";
        for (int j : vec[i]) {
          std::cout << j << ' ';
        }
        std::cout << '\n';
      }
      std::cout << "Time for GoogleNet Big concat: " << time << '\n';
    } else if (type == 3) {
      Graph graph1;
      std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                    891, 957,  1027, 973,  1008};
      size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
      int count_pic = static_cast<int>(sum) + 10;
      std::vector<float> res(count_pic * 28 * 28, 1.0F);
      Shape sh1({1, 5, 5, 3});
      std::vector<float> vec;
      vec.reserve(75);
      for (int i = 0; i < 75; ++i) {
        vec.push_back(3);
      }
      Tensor output = make_tensor(vec, sh1);

      Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
      Tensor t = make_tensor<float>(res, sh);
      input = t;
      build_graph_linear(graph1, input, output, options, false, false);
      Graph subgraph;
      std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
      std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("relu");
      subgraph.setInput(layer_0, input);
      subgraph.makeConnection(layer_0, layer_1);
      auto vec1 = find_subgraphs(graph1, subgraph);

      auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                       graph1, subgraph);
      for (int i = 0; i < vec1.size(); i++) {
        std::cout << i + 1 << ") ";
        for (int j : vec1[i]) {
          std::cout << j << ' ';
        }
        std::cout << '\n';
      }
      std::cout << "Time for AlexNet Conv -> ReLU: " << time << '\n';
    } else if (type == 4) {
      Graph graph1;
      build_graph(graph1, input, input, MODEL_PATH_YOLO11NET_ONNX, options,
                  false);

      Graph subgraph;
      std::shared_ptr<Layer> layer_0 = std::make_shared<ConvolutionalLayer>();
      std::shared_ptr<Layer> layer_1 = std::make_shared<EWLayer>("sigmoid");
      std::shared_ptr<Layer> layer_2 =
          std::make_shared<BinaryOpLayer>(BinaryOpLayer::Operation::kMul);
      subgraph.setInput(layer_0, input);
      subgraph.makeConnection(layer_0, layer_1);
      subgraph.makeConnection(layer_1, layer_2);
      subgraph.makeConnection(layer_0, layer_2);

      auto vec = find_subgraphs(graph1, subgraph);

      auto time = elapsed_time_avg<double, std::milli>(10, find_subgraphs,
                                                       graph1, subgraph);
      for (int i = 0; i < vec.size(); i++) {
        std::cout << i + 1 << ") ";
        for (int j : vec[i]) {
          std::cout << j << ' ';
        }
        std::cout << '\n';
      }
      std::cout << "Time for YoloNet (Conv -> Sigmoid, Conv) -> Mul: " << time << '\n';
    }
  }
  std::string temp;
  std::cin >> temp;
  return 0;
}
