#include "build.hpp"

void build_graph(Tensor& input, Tensor& output, bool comments,
                 bool parallel = false) {
  if (comments) {
    for (size_t i = 0; i < input.get_shape().dims(); i++) {
      std::cout << input.get_shape()[i] << ' ';
    }
    std::cout << std::endl;
    if (input.get_shape().dims() == 4) {
      for (size_t n = 0; n < input.get_shape()[0]; n++) {
        for (size_t h = 0; h < input.get_shape()[2]; h++) {
          for (size_t w = 0; w < input.get_shape()[3]; w++) {
            for (size_t c = 0; c < input.get_shape()[1]; c++) {
              std::cout << input.get<float>({n, c, h, w}) << ' ';
            }
          }
          std::cerr << std::endl;
        }
      }
      std::cout << std::endl << std::endl;
    }
  }
  ImplType impl1 = parallel ? kTBB : kDefault;
  ImplType impl2 = parallel ? kSTL : kDefault;
  std::vector<std::shared_ptr<Layer>> layers;

  std::string json_file = MODEL_PATH_H5;
  json model_data = read_json(json_file);

  if (comments) std::cout << "Loaded model data from JSON." << std::endl;

  for (const auto& layer_data : model_data) {
    std::string layer_type = layer_data["type"];
    if (comments)
      std::cout << "Processing layer of type: " << layer_type << std::endl;

    Tensor tensor = create_tensor_from_json(layer_data, Type::kFloat);

    if (layer_type.find("Conv") != std::string::npos) {
      Tensor tmp_tensor = tensor;
      // kernel is always transposed ?
      for (size_t n = 0; n < tensor.get_shape()[2]; n++) {
        for (size_t c = 0; c < tensor.get_shape()[3]; c++) {
          for (size_t h = 0; h < tensor.get_shape()[0]; h++) {
            for (size_t w = 0; w < tensor.get_shape()[1]; w++) {
              tmp_tensor.set<float>(std::vector<size_t>({w, h, n, c}),
                                    tensor.get<float>({h, w, n, c}));
            }
          }
        }
      }
      //
      tensor = tmp_tensor;
      Shape shape = tensor.get_shape();
      size_t pads = (tensor.get_shape()[0] - 1) / 2;
      if (layer_data.contains("padding")) {
        if (layer_data["padding"] == "valid") {
          pads = 0;
        }
      }
      if (comments) {
        std::cout << "PoolingLayer shape: ";
        for (size_t i = 0; i < shape.dims(); ++i) {
          std::cout << shape[i] << " ";
        }
        std::cout << std::endl;
      }

      Tensor tmp_values = tensor;
      Tensor tmp_bias = make_tensor(tensor.get_bias());
      auto conv_layer = std::make_shared<ConvolutionalLayer>(
          1, pads, 1, tmp_values, tmp_bias, impl2);
      conv_layer->setName(kConvolution);
      layers.push_back(conv_layer);
      if (comments) std::cout << "ConvLayer added to layers." << std::endl;
    }
    if (layer_type.find("relu") != std::string::npos) {
      auto ew_layer = std::make_shared<EWLayer>("relu");
      ew_layer->setName(kElementWise);
      layers.push_back(ew_layer);
      if (comments)
        std::cout << "Element wise (relu) added to layers" << std::endl;
    }
    if (layer_type.find("Dense") != std::string::npos) {
      Tensor tmp_bias = make_tensor(tensor.get_bias());
      Tensor tmp_tensor =
          Tensor(Shape({tensor.get_shape()[1], tensor.get_shape()[0]}),
                 itlab_2023::Type::kFloat);
      // kernel is always transposed ?
      for (size_t h = 0; h < tensor.get_shape()[0]; h++) {
        for (size_t w = 0; w < tensor.get_shape()[1]; w++) {
          tmp_tensor.set<float>(std::vector<size_t>({w, h}),
                                tensor.get<float>({h, w}));
        }
      }
      //
      tensor = tmp_tensor;
      auto fc_layer = std::make_shared<FCLayer>(tensor, tmp_bias);
      fc_layer->setName(kFullyConnected);
      layers.push_back(fc_layer);
      if (comments) std::cout << "DenseLayer added to layers." << std::endl;
    }

    if (layer_type.find("Pool") != std::string::npos) {
      Shape shape = {2, 2};
      std::string pooltype;
      if (layer_type.find("Max") != std::string::npos) {
        pooltype = "max";
      } else {
        pooltype = "average";
      }
      if (comments)
        std::cout << "PoolingLayer shape: " << shape[0] << "x" << shape[1]
                  << std::endl;
      auto pool_layer = std::make_shared<PoolingLayer>(shape, pooltype, impl1);
      pool_layer->setName(kPooling);
      layers.push_back(pool_layer);
      if (comments) std::cout << "PoolingLayer added to layers." << std::endl;
    }

    if (layer_type.find("Flatten") != std::string::npos) {
      auto flatten_layer =
          std::make_shared<FlattenLayer>(std::vector<size_t>({0, 3, 2, 1}));
      flatten_layer->setName(kFlatten);
      layers.push_back(flatten_layer);
      if (comments) std::cout << "FlattenLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dropout") != std::string::npos) {
      auto dropout_layer = std::make_shared<DropOutLayer>(0.0);
      dropout_layer->setName(kDropout);
      layers.push_back(dropout_layer);
      if (comments)
        std::cout
            << "DropOutLayer added to layers with probability 0.4 (turned "
               "off for inference)."
            << std::endl;
    }
  }
  if (comments)
    std::cout << "number of layers - " << layers.size() + 1 << std::endl;
  Graph graph(static_cast<int>(layers.size()));
  InputLayer a1(kNchw, kNchw);
  a1.setName(kInput);

  if (comments) std::cout << "InputLayer created." << std::endl;

  graph.setInput(a1, input);
  if (comments) std::cout << "Input set in graph." << std::endl;

  graph.makeConnection(a1, *layers[0]);
  if (comments)
    std::cout << "Connection made between InputLayer and first layer."
              << std::endl;

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    graph.makeConnection(*layers[i], *layers[i + 1]);
  }

  graph.setOutput(*layers.back(), output);
  if (comments) std::cout << "Output set in graph." << std::endl;

  if (comments) std::cout << "Starting inference..." << std::endl;
  graph.inference();
#ifdef ENABLE_STATISTIC_TIME
  std::vector<std::string> times = graph.getTimeInfo();
  std::cout << "!INFERENCE TIME INFO START!" << std::endl;
  for (size_t i = 0; i < times.size(); i++) {
    std::cout << times[i] << std::endl;
  }
  std::vector<int> elps_time = graph.getTime();
  int sum = std::accumulate(elps_time.begin(), elps_time.end(), 0);
  std::cout << "Elapsed inference time:" << sum << std::endl;
  std::cout << "!INFERENCE TIME INFO END!" << std::endl;
#endif
  if (comments) std::cout << "Inference completed." << std::endl;
  if (comments) {
    std::vector<float> tmp_output = softmax<float>(*output.as<float>());
    for (size_t i = 0; i < tmp_output.size(); i++) {
      if (tmp_output[i] < 1e-6) {
        std::cout << i << ": 0" << std::endl;
      } else {
        std::cout << i << ": " << tmp_output[i] << std::endl;
      }
    }
  }
}