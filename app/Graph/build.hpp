#pragma once
#include <functional>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <variant>
#include <vector>

#include "Weights_Reader/reader_weights.hpp"
#include "graph/graph.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/DropOutLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/FlattenLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"
#include "layers/Tensor.hpp"
#include "layers/ConcatLayer.hpp"
#include "layers/BinaryOpLayer.hpp"
#include "layers/SplitLayer.hpp"
#include "layers/TransposeLayer.hpp"
#include "layers/ReshapeLayer.hpp"
#include "layers/MatmulLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/ReduceLayer.hpp"
#include "layers/BatchNormalizationLayer.hpp"

void build_graph(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                 const std::string& json_path, bool comments,
                 bool parallel = false);
void build_graph_linear(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                 const std::string& json_path, bool comments,
                 bool parallel = false);