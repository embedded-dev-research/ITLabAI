#include "build.hpp"

void build_graph(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                 bool comments, bool parallel = false) {
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
  it_lab_ai::ImplType impl1 = parallel ? it_lab_ai::kTBB : it_lab_ai::kDefault;
  it_lab_ai::ImplType impl2 = parallel ? it_lab_ai::kSTL : it_lab_ai::kDefault;
  std::vector<std::shared_ptr<it_lab_ai::Layer>> layers;
  std::vector<bool> layerpostop;

  std::string json_file = MODEL_PATH_H5;
  it_lab_ai::json model_data = it_lab_ai::read_json(json_file);

  if (comments) std::cout << "Loaded model data from JSON." << std::endl;

  for (const auto& layer_data : model_data) {
    std::string layer_type = layer_data["type"];
    if (comments)
      std::cout << "Processing layer of type: " << layer_type << std::endl;

    it_lab_ai::Tensor tensor =
        it_lab_ai::create_tensor_from_json(layer_data, it_lab_ai::Type::kFloat);

    if (layer_type.find("Conv") != std::string::npos) {
      it_lab_ai::Tensor tmp_tensor = tensor;
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
      it_lab_ai::Shape shape = tensor.get_shape();
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

      it_lab_ai::Tensor tmp_values = tensor;
      it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());
      auto conv_layer = std::make_shared<it_lab_ai::ConvolutionalLayer>(
          1, pads, 1, tmp_values, tmp_bias, impl2);
      layers.push_back(conv_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "ConvLayer added to layers." << std::endl;
    }
    if (layer_type.find("relu") != std::string::npos) {
      auto ew_layer = std::make_shared<it_lab_ai::EWLayer>("relu");
      layers.push_back(ew_layer);
      layerpostop.push_back(true);
      if (comments)
        std::cout << "Element wise (relu) added to layers" << std::endl;
    }
    if (layer_type.find("Dense") != std::string::npos) {
      it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());
      it_lab_ai::Tensor tmp_tensor = it_lab_ai::Tensor(
          it_lab_ai::Shape({tensor.get_shape()[1], tensor.get_shape()[0]}),
          it_lab_ai::Type::kFloat);
      // kernel is always transposed ?
      for (size_t h = 0; h < tensor.get_shape()[0]; h++) {
        for (size_t w = 0; w < tensor.get_shape()[1]; w++) {
          tmp_tensor.set<float>(std::vector<size_t>({w, h}),
                                tensor.get<float>({h, w}));
        }
      }
      //
      tensor = tmp_tensor;
      auto fc_layer = std::make_shared<it_lab_ai::FCLayer>(tensor, tmp_bias);
      layers.push_back(fc_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "DenseLayer added to layers." << std::endl;
    }

    if (layer_type.find("Pool") != std::string::npos) {
      it_lab_ai::Shape shape = {2, 2};
      std::string pooltype;
      if (layer_type.find("Max") != std::string::npos) {
        pooltype = "max";
      } else {
        pooltype = "average";
      }
      if (comments)
        std::cout << "PoolingLayer shape: " << shape[0] << "x" << shape[1]
                  << std::endl;
      auto pool_layer =
          std::make_shared<it_lab_ai::PoolingLayer>(shape, pooltype, impl1);
      layers.push_back(pool_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "PoolingLayer added to layers." << std::endl;
    }

    if (layer_type.find("Flatten") != std::string::npos) {
      auto flatten_layer = std::make_shared<it_lab_ai::FlattenLayer>(
          std::vector<size_t>({0, 3, 2, 1}));
      layers.push_back(flatten_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "FlattenLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dropout") != std::string::npos) {
      auto dropout_layer = std::make_shared<it_lab_ai::DropOutLayer>(0.0);
      layers.push_back(dropout_layer);
      layerpostop.push_back(false);
      if (comments)
        std::cout
            << "DropOutLayer added to layers with probability 0.4 (turned "
               "off for inference)."
            << std::endl;
    }
  }
  if (comments)
    std::cout << "number of layers - " << layers.size() + 1 << std::endl;
  it_lab_ai::Graph graph(static_cast<int>(layers.size()));
  it_lab_ai::InputLayer a1(it_lab_ai::kNchw, it_lab_ai::kNchw);

  if (comments) std::cout << "InputLayer created." << std::endl;

  graph.setInput(a1, input);
  if (comments) std::cout << "Input set in graph." << std::endl;

  graph.makeConnection(a1, *layers[0]);
  if (comments)
    std::cout << "Connection made between InputLayer and first layer."
              << std::endl;

  for (size_t i = 0; i < layers.size() - 1; ++i) {
    if (layerpostop[i]) {
      layers[i - 1]->postops.layers.push_back(layers[i].get());
      layers[i - 1]->postops.count++;
      graph.makeConnection(*layers[i - 1], *layers[i + 1]);
    } else if (!layerpostop[i + 1])
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
    std::vector<float> tmp_output =
        it_lab_ai::softmax<float>(*output.as<float>());
    for (size_t i = 0; i < tmp_output.size(); i++) {
      if (tmp_output[i] < 1e-6) {
        std::cout << i << ": 0" << std::endl;
      } else {
        std::cout << i << ": " << tmp_output[i] << std::endl;
      }
    }
  }
}