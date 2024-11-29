#include "build.hpp"

void build_graph(Tensor input, Tensor output) {
  std::vector<std::shared_ptr<Layer>> layers;

  std::string json_file = MODEL_PATH;
  json model_data = read_json(json_file);

  std::cout << "Loaded model data from JSON." << std::endl;

  for (const auto& layer_data : model_data) {
    std::string layer_type = layer_data["type"];
    std::cout << "Processing layer of type: " << layer_type << std::endl;

    Tensor tensor =
        create_tensor_from_json(layer_data["weights"], Type::kFloat);

    if (layer_type.find("Conv") != std::string::npos) {
      Shape shape = tensor.get_shape();
      std::cout << "PoolingLayer shape: ";
      for (size_t i = 0; i < shape.dims(); ++i) {
        std::cout << shape[i] << " ";
      }
      std::cout << std::endl;

      Tensor tmp_values = tensor;
      Tensor tmp_bias = make_tensor(tensor.get_bias());

      auto conv_layer =
          std::make_shared<ConvolutionalLayer>(1, 0, 1, tmp_values, tmp_bias);
      conv_layer->setName(kConvolution);
      layers.push_back(conv_layer);
      std::cout << "ConvLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dense") != std::string::npos) {
      Tensor tmp_values = tensor;
      std::vector<float> Values_vector = *tensor.as<float>();
      std::vector<std::vector<float>> Values_vector_2d(
          tensor.get_shape()[0],
          std::vector<float>(tensor.get_shape()[1], 0.0f));
      int q = 0;
      for (int i = 0; i < Values_vector.size(); i++) {
        Values_vector_2d[q][i - (q * tensor.get_shape()[1])] = Values_vector[i];
        if ((i + 1) % tensor.get_shape()[1] == 0) {
          q++;
        }
      }
      std::vector<std::vector<float>> Values_vector_2d_2(
          tensor.get_shape()[1],
          std::vector<float>(tensor.get_shape()[0], 0.0f));

      for (int i = 0; i < tensor.get_shape()[0]; ++i) {
        for (int j = 0; j < tensor.get_shape()[1]; ++j) {
          Values_vector_2d_2[j][i] = Values_vector_2d[i][j];
        }
      }
      std::vector<float> Values_vector_1d(
          tensor.get_shape()[0] * tensor.get_shape()[1], 0.0f);
      int index_1d = 0;

      for (int j = 0; j < tensor.get_shape()[1]; ++j) {
        for (int k = 0; k < tensor.get_shape()[0]; ++k) {
          Values_vector_1d[index_1d++] = Values_vector_2d_2[j][k];
        }
      }

      Shape shape_fc({tensor.get_shape()[1], tensor.get_shape()[0]});
      Tensor values = make_tensor<float>(Values_vector_1d, shape_fc);
      Tensor tmp_bias = make_tensor(tensor.get_bias());

      auto fc_layer = std::make_shared<FCLayer>(values, tmp_bias);
      fc_layer->setName(kFullyConnected);
      layers.push_back(fc_layer);
      std::cout << "DenseLayer added to layers." << std::endl;
    }

    if (layer_type.find("Pool") != std::string::npos) {
      Shape shape = {2, 2};
      std::cout << "PoolingLayer shape: " << shape[0] << "x" << shape[1]
                << std::endl;
      auto pool_layer = std::make_shared<PoolingLayer>(shape);
      pool_layer->setName(kPooling);
      layers.push_back(pool_layer);
      std::cout << "PoolingLayer added to layers." << std::endl;
    }

    if (layer_type.find("Flatten") != std::string::npos) {
      auto flatten_layer = std::make_shared<FlattenLayer>();
      flatten_layer->setName(kFlatten);
      layers.push_back(flatten_layer);
      std::cout << "FlattenLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dropout") != std::string::npos) {
      auto dropout_layer = std::make_shared<DropOutLayer>(0.5);
      dropout_layer->setName(kDropout);
      layers.push_back(dropout_layer);
      std::cout << "DropOutLayer added to layers with probability 0.5."
                << std::endl;
    }
  }
  std::cout << "number of layers - " << layers.size() + 1 << std::endl;
  Graph graph(static_cast<int>(layers.size()));
  InputLayer a1(kNhwc, kNchw, 1, 2);

  std::cout << "InputLayer created." << std::endl;

  graph.setInput(a1, input);
  std::cout << "Input set in graph." << std::endl;

  graph.makeConnection(a1, *layers[0]);
  std::cout << "Connection made between InputLayer and first layer."
            << std::endl;

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    graph.makeConnection(*layers[i], *layers[i + 1]);
  }

  graph.setOutput(*layers.back(), output);
  std::cout << "Output set in graph." << std::endl;

  std::cout << "Starting inference..." << std::endl;
  graph.inference();
  std::cout << "Inference completed." << std::endl;

  std::vector<float> tmp = *output.as<float>();
  std::vector<float> tmp_output = softmax<float>(*output.as<float>());
  for (float i : tmp) {
    std::cout << i << " ";
  }
}