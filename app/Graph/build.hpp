#include <filesystem>
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

void build_graph(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                 bool comments, bool parallel);
