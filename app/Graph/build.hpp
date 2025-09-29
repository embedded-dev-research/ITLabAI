#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

#include "Weights_Reader/reader_weights.hpp"
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

void build_graph(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                 const std::string& json_path, bool comments,
                 bool parallel = false);
void build_graph_linear(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                        bool comments, bool parallel = false);
std::unordered_map<int, std::string> load_class_names(
    const std::string& filename);
std::unordered_map<std::string, std::string> model_paths = {
    {"alexnet_mnist", MODEL_PATH_H5},
    {"googlenet", MODEL_PATH_GOOGLENET_ONNX},
    {"resnet", MODEL_PATH_RESNET_ONNX},
    {"densenet", MODEL_PATH_DENSENET_ONNX},
    {"yolo", MODEL_PATH_YOLO11NET_ONNX}};
std::vector<int> get_input_shape_from_json(const std::string& json_path);
std::vector<float> process_model_output(const std::vector<float>& output,
                                        const std::string& model_name);
it_lab_ai::Tensor prepare_image(const cv::Mat& image,
                                const std::vector<int>& input_shape,
                                const std::string& model_name = "");
it_lab_ai::Tensor prepare_mnist_image(const cv::Mat& image);