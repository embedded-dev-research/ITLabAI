#pragma once
#include <tensorflow/c/c_api.h>
#include <graph/graph.hpp>
#include<tensorflow/c/tf_file_statistics.h>
#include <tensorflow/c/tf_status.h>
#include <tensorflow/c/tf_tensor.h>

#include <iostream>
Graph readTFModel(const std::string& modelPath, Graph& graph);
