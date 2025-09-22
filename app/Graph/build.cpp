#include "build.hpp"

#include <regex>
#include <set>
#include <unordered_map>
#include <unordered_set>

void build_graph_linear(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                        const std::string& json_path, bool comments,
                        bool parallel) {
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

  std::string json_file = json_path;
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
          1, pads, 1, tmp_values, tmp_bias, impl2, 1);
      conv_layer->setName(it_lab_ai::kConvolution);
      layers.push_back(conv_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "ConvLayer added to layers." << std::endl;
    }
    if (layer_type.find("relu") != std::string::npos) {
      auto ew_layer = std::make_shared<it_lab_ai::EWLayer>("relu");
      ew_layer->setName(it_lab_ai::kElementWise);
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
      fc_layer->setName(it_lab_ai::kFullyConnected);
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
      pool_layer->setName(it_lab_ai::kPooling);
      layers.push_back(pool_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "PoolingLayer added to layers." << std::endl;
    }

    if (layer_type.find("Flatten") != std::string::npos) {
      auto flatten_layer = std::make_shared<it_lab_ai::FlattenLayer>(
          std::vector<size_t>({0, 3, 2, 1}));
      flatten_layer->setName(it_lab_ai::kFlatten);
      layers.push_back(flatten_layer);
      layerpostop.push_back(false);
      if (comments) std::cout << "FlattenLayer added to layers." << std::endl;
    }

    if (layer_type.find("Dropout") != std::string::npos) {
      auto dropout_layer = std::make_shared<it_lab_ai::DropOutLayer>(0.0);
      dropout_layer->setName(it_lab_ai::kDropout);
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
  a1.setName(it_lab_ai::kInput);

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

std::string get_layer_name_by_id(
    const std::unordered_map<std::string, std::shared_ptr<it_lab_ai::Layer>>&
        name_to_layer,
    size_t layer_id) {
  for (const auto& [name, layer] : name_to_layer) {
    if (layer->getID() == layer_id) {
      return name;
    }
  }
  return "unknown_layer_" + std::to_string(layer_id);
}

std::string get_base_layer_name(const std::string& tensor_name) {
  std::regex pattern("(_output|_out|:)[_\\d]*$");
  return std::regex_replace(tensor_name, pattern, "");
}

std::string layerTypeToString(it_lab_ai::LayerType type) {
  switch (type) {
    case it_lab_ai::kInput:
      return "Input";
    case it_lab_ai::kPooling:
      return "Pooling";
    case it_lab_ai::kElementWise:
      return "ElementWise";
    case it_lab_ai::kConvolution:
      return "Convolution";
    case it_lab_ai::kFullyConnected:
      return "FullyConnected";
    case it_lab_ai::kFlatten:
      return "Flatten";
    case it_lab_ai::kConcat:
      return "Concat";
    case it_lab_ai::kDropout:
      return "Dropout";
    case it_lab_ai::kSplit:
      return "Split";
    case it_lab_ai::kBinaryOp:
      return "BinaryOp";
    case it_lab_ai::kTranspose:
      return "Transpose";
    case it_lab_ai::kMatmul:
      return "MatMul";
    case it_lab_ai::kReshape:
      return "Reshape";
    case it_lab_ai::kSoftmax:
      return "Softmax";
    case it_lab_ai::kReduce:
      return "Reduce";
    case it_lab_ai::kBatchNormalization:
      return "BatchNormalization";
    default:
      return "Unknown";
  }
}

void build_graph(it_lab_ai::Tensor& input, it_lab_ai::Tensor& output,
                 const std::string& json_path, bool comments, bool parallel) {
  /*if (comments) {
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
  }*/

  it_lab_ai::ImplType impl1 = parallel ? it_lab_ai::kTBB : it_lab_ai::kDefault;
  it_lab_ai::ImplType impl2 = parallel ? it_lab_ai::kSTL : it_lab_ai::kDefault;

  std::unordered_map<std::string, std::vector<std::string>> concat_connections;
  std::unordered_map<std::string, std::vector<int>> concat_orders;
  std::unordered_map<std::string, std::unordered_set<std::string>> concat_connected_inputs;

  std::unordered_map<std::string, std::vector<int64_t>> layer_parameters;
  std::unordered_map<std::string, float> float_parameters;
  std::string last_constant_name;
  std::vector<int64_t> last_constant_value;

  std::unordered_map<std::string, std::shared_ptr<it_lab_ai::SplitLayer>>
      split_layers;
  std::unordered_map<std::string, int> split_output_mapping;
  std::vector<std::vector<std::pair<int, int>>> split_distribution;
  std::unordered_map<std::string, int> split_name_to_index;
  std::unordered_map<std::string, int> original_ids;

  std::vector<std::shared_ptr<it_lab_ai::Layer>> layers;
  std::unordered_map<std::string, std::shared_ptr<it_lab_ai::Layer>>
      name_to_layer;
  std::unordered_map<std::string, std::vector<std::string>> connections;

  std::vector<std::pair<std::string, std::string>> connection_list;
  std::string json_file = json_path;

  it_lab_ai::json model_data = it_lab_ai::read_json(json_file);

  /*if (comments) std::cout << "Loaded model data from JSON." << std::endl;*/

  auto input_layer = std::make_shared<it_lab_ai::InputLayer>(it_lab_ai::kNchw,
                                                             it_lab_ai::kNchw);
  input_layer->setName(it_lab_ai::kInput);
  layers.push_back(input_layer);
  if (json_path == MODEL_PATH_RESNET_ONNX ||
      json_path == MODEL_PATH_DENSENET_ONNX) {
    name_to_layer["x"] = input_layer;
  } else if (json_path == MODEL_PATH_GOOGLENET_ONNX) {
    name_to_layer["image_tensor"] = input_layer;
  } else {
    name_to_layer["images"] = input_layer;
  }
  int current_id = 0;
  input_layer->setID(current_id++);
  for (const auto& layer_data : model_data) {
    try {
      std::string layer_type = layer_data["type"];

      if (layer_type == "InputLayer") continue;
      std::string layer_name = layer_data["name"];
      int layer_index = layer_data["index"];
      /*if (comments) {
        std::cout << "Processing layer " << layer_index << ": " << layer_name
                  << " (" << layer_type << ")" << std::endl;
      }*/

      std::shared_ptr<it_lab_ai::Layer> layer;

      if (layer_type.find("Conv") != std::string::npos) {
        it_lab_ai::Tensor tensor = it_lab_ai::create_tensor_from_json(
            layer_data, it_lab_ai::Type::kFloat);

        size_t stride = 1;
        size_t pads = 0;
        size_t group = 1;
        size_t dilations = 1;
        std::vector<size_t> pads_vec = {0, 0, 0, 0};

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];

          if (attributes.contains("strides") &&
              attributes["strides"].is_array()) {
            auto strides = attributes["strides"];
            if (strides.size() >= 2) {
              stride = strides[0].get<size_t>();
            }
          }

          if (attributes.contains("pads") && attributes["pads"].is_array()) {
            auto pads_array = attributes["pads"];
            if (pads_array.size() >= 4) {
              pads_vec = {
                  pads_array[0].get<size_t>(), pads_array[1].get<size_t>(),
                  pads_array[2].get<size_t>(), pads_array[3].get<size_t>()};
              pads = pads_vec[0];
            }
          } else if (layer_data.contains("padding") &&
                     layer_data["padding"] == "valid") {
            pads = 0;
          } else if (layer_data.contains("padding") &&
                     layer_data["padding"] == "same") {
            size_t kernel_size = tensor.get_shape()[0];
            pads = (kernel_size - 1) / 2;
          }

          if (attributes.contains("group")) {
            group = attributes["group"].get<size_t>();
          }

          if (attributes.contains("dilations") &&
              attributes["dilations"].is_array()) {
            auto dilations_array = attributes["dilations"];
            if (dilations_array.size() >= 2) {
              dilations = dilations_array[0].get<size_t>();
            }
          }
        }

        it_lab_ai::Tensor tmp_tensor = tensor;

        it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());

        auto conv_layer = std::make_shared<it_lab_ai::ConvolutionalLayer>(
            stride, pads, dilations, tmp_tensor, tmp_bias, impl2, group);
        conv_layer->setName(it_lab_ai::kConvolution);
        layer = conv_layer;
      } else if (layer_type.find("Relu") != std::string::npos ||
                 layer_type.find("relu") != std::string::npos) {
        auto ew_layer = std::make_shared<it_lab_ai::EWLayer>("relu");
        ew_layer->setName(it_lab_ai::kElementWise);
        layer = ew_layer;
      } else if (layer_type.find("Sigmoid") != std::string::npos) {
        auto ew_layer = std::make_shared<it_lab_ai::EWLayer>("sigmoid");
        ew_layer->setName(it_lab_ai::kElementWise);
        layer = ew_layer;

        /*if (comments) {
          std::cout << "Sigmoid layer added" << std::endl;
        }*/
      } else if (layer_type.find("Dense") != std::string::npos ||
                 layer_type.find("FullyConnected") != std::string::npos) {
        it_lab_ai::Tensor tensor = it_lab_ai::create_tensor_from_json(
            layer_data, it_lab_ai::Type::kFloat);

        it_lab_ai::Tensor tmp_tensor = it_lab_ai::Tensor(
            it_lab_ai::Shape({tensor.get_shape()[1], tensor.get_shape()[0]}),
            it_lab_ai::Type::kFloat);

        for (size_t h = 0; h < tensor.get_shape()[0]; h++) {
          for (size_t w = 0; w < tensor.get_shape()[1]; w++) {
            tmp_tensor.set<float>({w, h}, tensor.get<float>({h, w}));
          }
        }

        it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());
        auto fc_layer =
            std::make_shared<it_lab_ai::FCLayer>(tmp_tensor, tmp_bias);
        fc_layer->setName(it_lab_ai::kFullyConnected);
        layer = fc_layer;
      } else if (layer_type.find("Dropout") != std::string::npos) {
        auto dropout_layer = std::make_shared<it_lab_ai::DropOutLayer>(0.0);
        dropout_layer->setName(it_lab_ai::kDropout);
        layer = dropout_layer;
        if (comments)
          std::cout
              << "DropOutLayer added to layers with probability 0.4 (turned "
                 "off for inference)."
              << std::endl;
      } else if (layer_type == "GlobalAveragePool") {
        // Для GlobalAveragePool используем специальную логику
        auto pool_layer = std::make_shared<it_lab_ai::PoolingLayer>(
            it_lab_ai::Shape({0, 0}),  // special flag for global pooling
            "average", impl1);
        pool_layer->setName(it_lab_ai::kPooling);
        layer = pool_layer;

        /*if (comments) {
          std::cout << "GlobalAveragePool layer added (will use input spatial "
                       "dimensions as kernel)"
                    << std::endl;
        }*/
      } else if ((layer_type == "MaxPool" || layer_type == "AveragePool")) {
        std::string pooltype =
            (layer_type.find("Max") != std::string::npos) ? "max" : "average";

        it_lab_ai::Shape shape = {2, 2};
        it_lab_ai::Shape strides = {2, 2};
        it_lab_ai::Shape pads = {0, 0, 0, 0};
        it_lab_ai::Shape dilations = {1, 1};
        bool ceil_mode = false;

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];

          if (attributes.contains("kernel_shape") &&
              attributes["kernel_shape"].is_array()) {
            auto kernel_shape = attributes["kernel_shape"];
            if (kernel_shape.size() >= 2) {
              shape = it_lab_ai::Shape({kernel_shape[0].get<size_t>(),
                                        kernel_shape[1].get<size_t>()});
            }
          }

          if (attributes.contains("strides") &&
              attributes["strides"].is_array()) {
            auto strides_array = attributes["strides"];
            if (strides_array.size() >= 2) {
              strides = it_lab_ai::Shape({strides_array[0].get<size_t>(),
                                          strides_array[1].get<size_t>()});
            }
          }

          if (attributes.contains("pads") && attributes["pads"].is_array()) {
            auto pads_array = attributes["pads"];
            if (pads_array.size() >= 4) {
              pads = it_lab_ai::Shape(
                  {pads_array[0].get<size_t>(), pads_array[1].get<size_t>(),
                   pads_array[2].get<size_t>(), pads_array[3].get<size_t>()});
            }
          }

          if (attributes.contains("dilations") &&
              attributes["dilations"].is_array()) {
            auto dilations_array = attributes["dilations"];
            if (dilations_array.size() >= 2) {
              dilations = it_lab_ai::Shape({dilations_array[0].get<size_t>(),
                                            dilations_array[1].get<size_t>()});
            }
          }

          if (attributes.contains("ceil_mode")) {
            ceil_mode = attributes["ceil_mode"].get<int>() != 0;
          }
        }

        auto pool_layer =
            std::make_shared<it_lab_ai::PoolingLayer>(shape, pooltype, impl1);

        try {
          if (strides[0] != 2 || strides[1] != 2) {
            pool_layer->setStrides(strides[0], strides[1]);
          }

          if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
            pool_layer->setPads(pads[0], pads[1], pads[2], pads[3]);
          }

          if (dilations[0] != 1 || dilations[1] != 1) {
            pool_layer->setDilations(dilations[0], dilations[1]);
          }

          pool_layer->setCeilMode(ceil_mode);

        } catch (const std::exception& e) {
          if (comments) {
            std::cout << "Warning: Some pooling parameters not supported: "
                      << e.what() << std::endl;
          }
        }

        pool_layer->setName(it_lab_ai::kPooling);
        layer = pool_layer;
      } else if (layer_type.find("Flatten") != std::string::npos) {
        int axis = 1;  // значение по умолчанию

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("axis")) {
            axis = attributes["axis"].get<int>();
          }
        }

        /*if (comments) {
          std::cout << "Flatten layer with axis: " << axis << std::endl;
        }*/

        auto flatten_layer = std::make_shared<it_lab_ai::FlattenLayer>(axis);
        flatten_layer->setName(it_lab_ai::kFlatten);
        layer = flatten_layer;
      } else if (layer_type == "Concat") {
        int axis = 0;
        if (layer_data["attributes"].contains("axis")) {
          axis = layer_data["attributes"]["axis"];
        }
        if (layer_data.contains("inputs")) {
          for (const auto& input_name : layer_data["inputs"]) {
            std::string input_tensor = input_name.get<std::string>();
            std::string base_input_name = get_base_layer_name(input_tensor);
            concat_connections[layer_name].push_back(base_input_name);
          }
        }
        auto concat_layer = std::make_shared<it_lab_ai::ConcatLayer>(axis);
        concat_layer->setName(it_lab_ai::kConcat);
        layer = concat_layer;
        concat_connected_inputs[layer_name] = std::unordered_set<std::string>();
      } else if (layer_type == "Split") {
        int axis = 0;
        std::vector<int64_t> splits;
        size_t num_outputs = 2;

        if (layer_data["attributes"].contains("axis")) {
          axis = layer_data["attributes"]["axis"];
        }
        if (layer_data.contains("inputs") && layer_data["inputs"].is_array()) {
          auto inputs = layer_data["inputs"];
          if (inputs.size() >= 2) {
            std::string constant_name = inputs[1].get<std::string>();
            constant_name = get_base_layer_name(constant_name);

            if (layer_parameters.count(constant_name)) {
              splits = layer_parameters[constant_name];
            } else if (constant_name.find("onnx::") != constant_name.npos) {
              splits = last_constant_value;
              layer_parameters[constant_name] = last_constant_value;
            }
          }
        }
        if (layer_data.contains("weights") &&
            layer_data["weights"].is_array()) {
          for (const auto& s : layer_data["weights"]) {
            splits.push_back(s.get<int>());
          }
          num_outputs = splits.size();
        }

        auto split_layer = std::make_shared<it_lab_ai::SplitLayer>(
            static_cast<int>(axis), splits);
        split_layer->setName(it_lab_ai::kSplit);
        layer = split_layer;

        // Сохраняем сплит-слой для последующего использования
        split_layers[layer_name] = split_layer;
        split_name_to_index[layer_name] =
            static_cast<int>(split_distribution.size());
        // Создаем запись в распределении для этого сплита
        split_distribution.emplace_back();
      } else if (layer_type == "Add" || layer_type == "Mul" ||
                 layer_type == "Sub" || layer_type == "Div") {
        // Проверяем, есть ли среди входов СКАЛЯРНЫЕ константы
        bool has_scalar_constant = false;
        float scalar_value = 0.0f;

        if (layer_data.contains("inputs") && layer_data["inputs"].is_array()) {
          auto inputs = layer_data["inputs"];
          for (const auto& input_name : inputs) {
            std::string input_tensor = input_name.get<std::string>();
            std::string base_name = get_base_layer_name(input_tensor);

            // Ищем скалярные константы (они сохраняются в
            // layer_parameters/float_parameters)
            if (float_parameters.find(base_name) != float_parameters.end()) {
              scalar_value = float_parameters[base_name];
              has_scalar_constant = true;
              break;
            } else if (layer_parameters.find(base_name) !=
                           layer_parameters.end() &&
                       !layer_parameters[base_name].empty()) {
              scalar_value = static_cast<float>(layer_parameters[base_name][0]);
              has_scalar_constant = true;
              break;
            }
          }
        }

        // Проверяем прямое значение value
        bool has_direct_value = layer_data.contains("value");
        float direct_value = 0.0f;

        if (has_direct_value) {
          if (layer_data["value"].is_string()) {
            try {
              direct_value = std::stof(layer_data["value"].get<std::string>());
            } catch (...) {
              direct_value = 0.0f;
            }
          } else if (layer_data["value"].is_number()) {
            direct_value = layer_data["value"].get<float>();
          }
        }

        // Унарная операция ТОЛЬКО если есть скалярное значение
        if (has_direct_value || has_scalar_constant) {
          float value = has_direct_value ? direct_value : scalar_value;
          std::string ew_operation;

          if (layer_type == "Mul") {
            ew_operation = "linear";
            auto ew_layer =
                std::make_shared<it_lab_ai::EWLayer>(ew_operation, value, 0.0f);
            ew_layer->setName(it_lab_ai::kElementWise);
            layer = ew_layer;
            /*if (comments) {
              std::cout << "Created binary " << layer_type << " operation with "
                        << value <<"scalar" << std::endl;
            }*/
          } else if (layer_type == "Add") {
            ew_operation = "linear";
            auto ew_layer =
                std::make_shared<it_lab_ai::EWLayer>(ew_operation, 1.0f, value);
            ew_layer->setName(it_lab_ai::kElementWise);
            layer = ew_layer;
          } else if (layer_type == "Sub") {
            ew_operation = "linear";
            auto ew_layer = std::make_shared<it_lab_ai::EWLayer>(ew_operation,
                                                                 1.0f, -value);
            ew_layer->setName(it_lab_ai::kElementWise);
            layer = ew_layer;
          } else {
            if (comments) {
              std::cout << "Unsupported unary operation: " << layer_type
                        << " with value, skipping..." << std::endl;
            }
            continue;
          }
        } else {
          // Бинарная операция (два тензора) - ЭТО ВАШ СЛУЧАЙ!
          it_lab_ai::BinaryOpLayer::Operation op;
          if (layer_type == "Add")
            op = it_lab_ai::BinaryOpLayer::Operation::kAdd;
          else if (layer_type == "Sub")
            op = it_lab_ai::BinaryOpLayer::Operation::kSub;
          else if (layer_type == "Mul")
            op = it_lab_ai::BinaryOpLayer::Operation::kMul;
          else if (layer_type == "Div")
            op = it_lab_ai::BinaryOpLayer::Operation::kDiv;

          auto bin_layer = std::make_shared<it_lab_ai::BinaryOpLayer>(op);
          bin_layer->setName(it_lab_ai::kBinaryOp);
          layer = bin_layer;
          
        }
      } else if (layer_type == "Gemm") {
        it_lab_ai::Tensor tensor = it_lab_ai::create_tensor_from_json(
            layer_data, it_lab_ai::Type::kFloat);

        float alpha = 1.0f;
        float beta = 1.0f;
        bool transB = true;

        if (layer_data.contains("alpha")) {
          alpha = layer_data["alpha"].get<float>();
        }
        if (layer_data.contains("beta")) {
          beta = layer_data["beta"].get<float>();
        }
        if (layer_data.contains("transB")) {
          transB = layer_data["transB"].get<int>() != 0;
        }

        it_lab_ai::Tensor tmp_tensor = tensor;
        

        // Bias остается оригинальным (НЕ меняется при транспонировании weights)
        it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());
        if (transB) {
          // Создаем тензор с транспонированной формой
          it_lab_ai::Shape transposed_shape(
              {tensor.get_shape()[1], tensor.get_shape()[0]});
          it_lab_ai::Tensor transposed_tensor(transposed_shape,
                                              it_lab_ai::Type::kFloat);

          // Транспонируем данные
          for (size_t i = 0; i < tensor.get_shape()[0]; ++i) {
            for (size_t j = 0; j < tensor.get_shape()[1]; ++j) {
              float value = tensor.get<float>({i, j});
              transposed_tensor.set<float>({j, i}, value);
            }
          }

          tmp_tensor = transposed_tensor;

          /*if (comments) {
            std::cout << "Weights transposed from [" << tensor.get_shape()[0]
                      << ", " << tensor.get_shape()[1] << "] to ["
                      << transposed_shape[0] << ", " << transposed_shape[1]
                      << "]" << std::endl;
          }*/
        }

        // Применяем alpha к весам
        if (alpha != 1.0f) {
          auto weights_data = *tmp_tensor.as<float>();
          for (auto& val : weights_data) {
            val *= alpha;
          }
          tmp_tensor = make_tensor(weights_data, tmp_tensor.get_shape());
        }

        // Применяем beta к bias
        if (beta != 1.0f) {
          auto bias_data = *tmp_bias.as<float>();
          for (auto& val : bias_data) {
            val *= beta;
          }
          tmp_bias = make_tensor(bias_data, tmp_bias.get_shape());
        }

        auto fc_layer =
            std::make_shared<it_lab_ai::FCLayer>(tmp_tensor, tmp_bias);
        fc_layer->setName(it_lab_ai::kFullyConnected);
        layer = fc_layer;
      } else if (layer_type == "Transpose" ||
                 layer_type.find("transpose") != std::string::npos) {
        std::vector<int64_t> perm;
        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("perm") && attributes["perm"].is_array()) {
            auto perm_array = attributes["perm"];
            for (const auto& p : perm_array) {
              perm.push_back(p.get<int64_t>());
            }
          }
        }

        auto transpose_layer =
            std::make_shared<it_lab_ai::TransposeLayer>(perm);
        transpose_layer->setName(it_lab_ai::kTranspose);
        layer = transpose_layer;

        /*if (comments) {
          std::cout << "TransposeLayer added with perm: [";
          for (size_t i = 0; i < perm.size(); ++i) {
            std::cout << perm[i];
            if (i < perm.size() - 1) std::cout << ", ";
          }
          std::cout << "]" << std::endl;
        }*/
      } else if (layer_type == "Reshape") {
        bool allowzero = false;
        std::vector<int64_t> shape;

        if (layer_data.contains("inputs") && layer_data["inputs"].is_array()) {
          auto inputs = layer_data["inputs"];
          if (inputs.size() >= 2) {
            std::string constant_name = inputs[1].get<std::string>();
            constant_name = get_base_layer_name(constant_name);

            if (layer_parameters.count(constant_name)) {
              shape = layer_parameters[constant_name];
            }
          }
        }

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("allowzero")) {
            allowzero = attributes["allowzero"].get<int64_t>() != 0;
          }
        }

        if (layer_data.contains("weights") &&
            layer_data["weights"].is_array()) {
          auto weights = layer_data["weights"];
          for (const auto& weight : weights) {
            if (weight.is_number()) {
              shape.push_back(weight.get<int64_t>());
            }
          }
        }

        auto reshape_layer =
            std::make_shared<it_lab_ai::ReshapeLayer>(allowzero, shape);
        reshape_layer->setName(it_lab_ai::kReshape);
        layer = reshape_layer;

        /*if (comments) {
          std::cout << "ReshapeLayer added with allowzero: " << allowzero;
          if (!shape.empty()) {
            std::cout << ", shape: [";
            for (size_t i = 0; i < shape.size(); ++i) {
              std::cout << shape[i];
              if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
          }
          std::cout << std::endl;
        }*/
      } else if (layer_type == "ReduceMean") {
        /*if (comments) {
          std::cout << "ReduceMean layer: " << layer_name << std::endl;
        }*/

        std::vector<int64_t> axes;
        int64_t keepdims = 1;

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("axes") && attributes["axes"].is_array()) {
            auto axes_array = attributes["axes"];
            for (const auto& axis : axes_array) {
              axes.push_back(axis.get<int64_t>());
            }
          }
          if (attributes.contains("keepdims")) {
            keepdims = attributes["keepdims"].get<int64_t>();
          }
        }
        auto reduce_layer = std::make_shared<it_lab_ai::ReduceLayer>(
            it_lab_ai::ReduceLayer::Operation::kMean, keepdims, axes);
        reduce_layer->setName(it_lab_ai::kReduce);
        layer = reduce_layer;
      } else if (layer_type == "ReduceSum") {
        /*if (comments) {
          std::cout << "ReduceSum layer: " << layer_name << std::endl;
        }*/
        int64_t keepdims = 0;
        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("keepdims")) {
            keepdims = attributes["keepdims"].get<int64_t>();
          }
        }

        std::vector<int64_t> axes;
        if (layer_data.contains("inputs") && layer_data["inputs"].is_array()) {
          auto inputs = layer_data["inputs"];
          if (inputs.size() >= 2) {
            std::string constant_name = inputs[1].get<std::string>();
            constant_name = get_base_layer_name(constant_name);

            if (layer_parameters.count(constant_name)){
              axes = layer_parameters[constant_name];
            } else if (constant_name.find("onnx::") != constant_name.npos) {
              axes = last_constant_value;
              layer_parameters[constant_name] = last_constant_value;
            }
          }
        }
        auto reduce_layer = std::make_shared<it_lab_ai::ReduceLayer>(
            it_lab_ai::ReduceLayer::Operation::kSum, keepdims, axes);
        reduce_layer->setName(it_lab_ai::kReduce);
        layer = reduce_layer;
      } else if (layer_type == "Constant") {
        /*if (comments) {
          std::cout << "Constant layer: " << layer_name << std::endl;
        }*/

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("value") && attributes["value"].is_array()) {
            auto values = attributes["value"];
            std::vector<int64_t> data;
            for (const auto& val : values) {
              data.push_back(val.get<int64_t>());
            }
            layer_parameters[layer_name] = data;
            last_constant_name = layer_name;
            last_constant_value = data;
          }
          if (attributes.contains("value") && attributes["value"].is_number()) {
            float value = attributes["value"].get<float>();
            float_parameters[layer_name] = value;
          }
        }

        continue;
      } else if (layer_type == "MatMul") {
        auto matmul_layer = std::make_shared<it_lab_ai::MatmulLayer>();
        matmul_layer->setName(it_lab_ai::kMatmul);
        layer = matmul_layer;

        /*if (comments) {
          std::cout << "MatMul layer added" << std::endl;
        }*/
      } else if (layer_type == "Softmax") {
        int axis = -1;

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("axis")) {
            axis = attributes["axis"].get<int>();
          }
        }

        auto softmax_layer = std::make_shared<it_lab_ai::SoftmaxLayer>(axis);
        softmax_layer->setName(it_lab_ai::kSoftmax);
        layer = softmax_layer;

        /*if (comments) {
          std::cout << "Softmax layer added with axis: " << axis << std::endl;
        }*/
      } else if (layer_type == "BatchNormalization") {
        /*if (comments) {
          std::cout << "BatchNormalization layer: " << layer_name << std::endl;
        }*/

        float epsilon = 1e-5f;
        float momentum = 0.9f;
        bool training_mode = false;

        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];
          if (attributes.contains("epsilon")) {
            epsilon = attributes["epsilon"].get<float>();
          }
          if (attributes.contains("momentum")) {
            momentum = attributes["momentum"].get<float>();
          }
          if (attributes.contains("training_mode")) {
            training_mode = attributes["training_mode"].get<int64_t>() != 0;
          }
        }

        std::vector<float> scale_data;
        std::vector<float> bias_data;
        std::vector<float> mean_data;
        std::vector<float> var_data;

        if (layer_data.contains("scale") && layer_data["scale"].is_array()) {
          const auto& scale_array = layer_data["scale"];
          for (const auto& value : scale_array) {
            scale_data.push_back(value.get<float>());
          }
        }

        if (layer_data.contains("bias") && layer_data["bias"].is_array()) {
          const auto& bias_array = layer_data["bias"];
          for (const auto& value : bias_array) {
            bias_data.push_back(value.get<float>());
          }
        }

        if (layer_data.contains("mean") && layer_data["mean"].is_array()) {
          const auto& mean_array = layer_data["mean"];
          for (const auto& value : mean_array) {
            mean_data.push_back(value.get<float>());
          }
        }

        if (layer_data.contains("var") && layer_data["var"].is_array()) {
          const auto& var_array = layer_data["var"];
          for (const auto& value : var_array) {
            var_data.push_back(value.get<float>());
          }
        }

        size_t num_channels = scale_data.size();

        it_lab_ai::Tensor scale = it_lab_ai::make_tensor(
            scale_data, it_lab_ai::Shape({num_channels}));
        it_lab_ai::Tensor bias =
            it_lab_ai::make_tensor(bias_data, it_lab_ai::Shape({num_channels}));
        it_lab_ai::Tensor mean =
            it_lab_ai::make_tensor(mean_data, it_lab_ai::Shape({num_channels}));
        it_lab_ai::Tensor var =
            it_lab_ai::make_tensor(var_data, it_lab_ai::Shape({num_channels}));

        auto bn_layer = std::make_shared<it_lab_ai::BatchNormalizationLayer>(
            scale, bias, mean, var, epsilon, momentum, training_mode);
        bn_layer->setName(it_lab_ai::kBatchNormalization);
        layer = bn_layer;
      } else {
        /*if (comments) {
          std::cout << "Warning: Unknown layer type: " << layer_type
                    << std::endl;
        }*/
        continue;
      }
      if (layer) {
        int original_id = current_id;
        layer->setID(current_id++);
        layers.push_back(layer);
        name_to_layer[layer_name] = layer;
        original_ids[layer_name] = original_id;
        if (layer_data.contains("inputs")) {
          for (const auto& input_name : layer_data["inputs"]) {
            std::string input_tensor = input_name.get<std::string>();

            

            // Проверяем, является ли вход выходом сплит-слоя
            std::regex split_output_pattern("(.+)_output_(\\d+)$");
            std::smatch matches;

            if (std::regex_search(input_tensor, matches,
                                  split_output_pattern)) {
              std::string split_layer_name = matches[1].str();
              int output_index = std::stoi(matches[2].str());

              if (split_layers.find(split_layer_name) != split_layers.end()) {
                // Это выход сплит-слоя, добавляем в распределение
                int split_layer_id = split_layers[split_layer_name]->getID();
                int target_layer_id = layer->getID();

                // Находим индекс этого сплита в split_distribution
                int split_index = split_name_to_index[split_layer_name];

                // Проверяем, нет ли уже такого соединения в распределении
                bool connection_exists = false;
                for (const auto& existing_conn :
                     split_distribution[split_index]) {
                  if (existing_conn.first == target_layer_id &&
                      existing_conn.second == output_index) {
                    connection_exists = true;
                    break;
                  }
                }

                if (!connection_exists) {
                  // Добавляем связь в распределение сплитов: (ID целевого слоя,
                  // индекс выхода сплита)
                  split_distribution[split_index].emplace_back(target_layer_id,
                                                               output_index);
                }

                // ТАКЖЕ добавляем обычное соединение в connections (только если
                // его еще нет)
                bool connection_in_list = false;
                for (const auto& existing_target :
                     connections[split_layer_name]) {
                  if (existing_target == layer_name) {
                    connection_in_list = true;
                    break;
                  }
                }

                if (!connection_in_list) {
                  connections[split_layer_name].push_back(layer_name);
                }

                /*if (comments) {
                  std::cout << "Split connection: " << split_layer_name
                            << " output " << output_index << " -> "
                            << layer_name << " (split index: " << split_index
                            << ", target ID: " << target_layer_id << ")";
                  if (connection_exists) {
                    std::cout << " [ALREADY EXISTS IN DISTRIBUTION]";
                  }
                  if (connection_in_list) {
                    std::cout << " [ALREADY EXISTS IN CONNECTIONS]";
                  }
                  std::cout << std::endl;
                }*/

                continue;  // Пропускаем дальнейшую обработку этого входа
              }
            }

            // Обычная обработка не-сплит соединений
            if (input_tensor.find("Constant") != std::string::npos ||
                input_tensor.find("onnx::") != std::string::npos ||
                input_tensor.find("_Constant") != std::string::npos) {
              /*if (comments) {
                std::cout << "Skipping connection from: " << input_tensor
                          << std::endl;
              }*/
              continue;
            }
            connections[input_tensor].push_back(layer_name);
          }
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Error processing layer " << layer_data["index"] << " ("
                << layer_data["name"] << "): " << e.what() << std::endl;
      throw;
    }
  }

  /*if (comments) {
    std::cout << "\n=== name_to_layer CONTENTS ===" << std::endl;
    std::cout << "Total layers in name_to_layer: " << name_to_layer.size()
              << std::endl;
    for (const auto& [name, layer_ptr] : name_to_layer) {
      std::cout << "  '" << name << "' -> ID: " << layer_ptr->getID()
                << ", Type: " << layerTypeToString(layer_ptr->getName())
                << std::endl;
    }

    std::cout << "\n=== connections CONTENTS ===" << std::endl;
    std::cout << "Total connections: " << connections.size() << std::endl;
    for (const auto& [source_name, target_names] : connections) {
      std::cout << "  '" << source_name << "' -> ";
      for (const auto& target_name : target_names) {
        std::cout << "'" << target_name << "' ";
      }
      std::cout << std::endl;
    }
  }*/

  // После заполнения split_distribution, перед созданием графа добавим
  // отладочный вывод
  /*if (comments) {
    std::cout << "\n=== SPLIT DISTRIBUTION ===" << std::endl;
    std::cout << "Total splits: " << split_distribution.size() << std::endl;

    for (size_t i = 0; i < split_distribution.size(); i++) {
      std::cout << "Split " << i << " connections: ";
      if (split_distribution[i].empty()) {
        std::cout << "EMPTY";
      } else {
        for (const auto& conn : split_distribution[i]) {
          std::cout << "(" << conn.first << ", output " << conn.second << ") ";
        }
      }
      std::cout << std::endl;
    }
  }*/

  it_lab_ai::Graph graph(static_cast<int>(layers.size()));

  graph.setInput(*input_layer, input);

  /*if (comments) {
    std::cout << "\n=== CREATING GRAPH CONNECTIONS ===" << std::endl;
  }*/
  for (const auto& [source_tensor, target_layers] : connections) {
    std::string source_layer_name = get_base_layer_name(source_tensor);

    for (const auto& target_layer_name : target_layers) {
      connection_list.emplace_back(source_layer_name, target_layer_name);
      /*if (comments) {
        std::cout << "Planned connection: " << source_layer_name << " -> "
                  << target_layer_name << std::endl;
      }*/
    }
  }

  /*if (comments) {
    std::cout << "\n=== BEFORE SORTING CONNECTIONS ===" << std::endl;
    for (const auto& conn : connection_list) {
      std::cout << "Connection: " << conn.first << " -> " << conn.second;
      if (name_to_layer.count(conn.first)) {
        std::cout << " (ID: " << name_to_layer[conn.first]->getID() << ")";
      } else {
        std::cout << " (SOURCE NOT FOUND!)";
      }
      if (name_to_layer.count(conn.second)) {
        std::cout << " -> (ID: " << name_to_layer[conn.second]->getID() << ")";
      } else {
        std::cout << " -> (TARGET NOT FOUND!)";
      }
      std::cout << std::endl;
    }
  }*/

  try {
    std::sort(
        connection_list.begin(), connection_list.end(),
        [&](const auto& a, const auto& b) {
          if (!name_to_layer.count(a.first) || !name_to_layer.count(b.first)) {
            return false;
          }
          return name_to_layer[a.first]->getID() <
                 name_to_layer[b.first]->getID();
        });

    /*if (comments) {
      std::cout << "Sorting completed successfully" << std::endl;
    }*/
  } catch (const std::exception& e) {
    std::cerr << "ERROR during sorting: " << e.what() << std::endl;
  }

  /*if (comments) {
    std::cout << "\n=== ESTABLISHING CONNECTIONS ===" << std::endl;
  }*/
  std::vector<int> order = {};

  for (const auto& [source_name, target_name] : connection_list) {
    if (name_to_layer.count(source_name) && name_to_layer.count(target_name)) {
      // Обработка Concat слоев
      if (target_name.find("Concat") != std::string::npos ||
          name_to_layer[target_name]->getName() == it_lab_ai::kConcat) {
        // Проверяем, есть ли этот concat в нашем списке
        if (concat_connections.find(target_name) != concat_connections.end()) {
          // Находим индекс этого источника в ожидаемых входах concat
          const auto& expected_inputs = concat_connections[target_name];
          auto it = std::find(expected_inputs.begin(), expected_inputs.end(),
                              source_name);

          if (it != expected_inputs.end()) {
            int input_index = static_cast<int>(std::distance(expected_inputs.begin(), it));

            // Добавляем индекс в порядок для этого concat
            concat_orders[target_name].push_back(input_index);

            // Отмечаем, что этот вход подключен
            concat_connected_inputs[target_name].insert(source_name);

            if (comments) {
              std::cout << "Concat connection: " << source_name << " -> "
                        << target_name << " (index: " << input_index << ")"
                        << std::endl;
            }

            // Проверяем, все ли входы подключены
            if (concat_connected_inputs[target_name].size() ==
                concat_connections[target_name].size()) {
              // Все входы подключены - устанавливаем порядок
              auto concat_layer =
                  std::dynamic_pointer_cast<it_lab_ai::ConcatLayer>(
                      name_to_layer[target_name]);
              if (concat_layer) {
                concat_layer->setInputOrder(concat_orders[target_name]);

                if (comments) {
                  std::cout
                      << "=== ALL INPUTS CONNECTED TO CONCAT: " << target_name
                      << " ===" << std::endl;
                  std::cout << "Expected inputs: ";
                  for (const auto& inp : concat_connections[target_name]) {
                    std::cout << inp << " ";
                  }
                  std::cout << std::endl;

                  std::cout << "Actual order: ";
                  for (size_t i = 0; i < concat_orders[target_name].size();
                       ++i) {
                    std::cout << concat_orders[target_name][i];
                    if (i < concat_orders[target_name].size() - 1)
                      std::cout << ", ";
                  }
                  std::cout << std::endl;
                }
              }
            }
          }
        }
      }

      try {
        graph.makeConnection(*name_to_layer[source_name],
                             *name_to_layer[target_name]);

      } catch (const std::exception& e) {
        std::cerr << "Failed: " << source_name << " -> " << target_name << " : "
                  << e.what() << std::endl;
      }
    }
  }
  for (auto& split_dist : split_distribution) {
    for (auto& connection : split_dist) {
      // Находим слой по старому ID и получаем его новый ID
      for (const auto& [name, layer] : name_to_layer) {
        if (original_ids[name] == connection.first) {
          connection.first = layer->getID();
          break;
        }
      }
    }
  }
  graph.setSplitDistribution(split_distribution);
  auto output_layer = layers.back();
  graph.setOutput(*output_layer, output);
  auto in_out_degrees = graph.getInOutDegrees();
  auto traversal_order = graph.getTraversalOrder();

    if (comments) std::cout << "Starting inference..." << std::endl;
    try {
      graph.inference();
      if (comments)
        std::cout << "Inference completed successfully." << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "ERROR during inference: " << e.what() << std::endl;
    }

//#ifdef ENABLE_STATISTIC_TIME
//    std::vector<std::string> times = graph.getTimeInfo();
//    std::cout << "!INFERENCE TIME INFO START!" << std::endl;
//    for (size_t i = 0; i < times.size(); i++) {
//      std::cout << times[i] << std::endl;
//    }
//    std::vector<int> elps_time = graph.getTime();
//    int sum = std::accumulate(elps_time.begin(), elps_time.end(), 0);
//    std::cout << "Elapsed inference time:" << sum << std::endl;
//    std::cout << "!INFERENCE TIME INFO END!" << std::endl;
//#endif
//
    //if (comments) {
    //  try {
    //    std::cout << "Output tensor size: " << output.get_shape().count()
    //              << std::endl;
    //    std::vector<float> tmp_output =
    //        it_lab_ai::softmax<float>(*output.as<float>());
    //    std::cout << "Softmax completed, showing top 5 classes:" << std::endl;

    //    // Выводите только топ-5 классов вместо всех 1000
    //    std::vector<size_t> indices(tmp_output.size());
    //    std::iota(indices.begin(), indices.end(), 0);
    //    std::partial_sort(
    //        indices.begin(), indices.begin() + 5, indices.end(),
    //        [&](size_t a, size_t b) { return tmp_output[a] > tmp_output[b]; });

    //    for (int i = 0; i < 5; i++) {
    //      size_t idx = indices[i];
    //      std::cout << "Class " << idx << ": " << tmp_output[idx] << std::endl;
    //    }
    //  } catch (const std::exception& e) {
    //    std::cerr << "Error in output processing: " << e.what() << std::endl;
    //  }
    //}
}

