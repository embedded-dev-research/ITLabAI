#include "build.hpp"
#include <regex>
#include <set>
#include <unordered_map>

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
    case it_lab_ai::kSplit:
      return "Split";
    case it_lab_ai::kBinaryOp:
      return "BinaryOp";
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

  std::vector<std::shared_ptr<it_lab_ai::Layer>> layers;
  std::unordered_map<std::string, std::shared_ptr<it_lab_ai::Layer>>
      name_to_layer;
  std::unordered_map<std::string, std::vector<std::string>> connections;

  std::vector<std::pair<std::string, std::string>> connection_list;
  std::string json_file = json_path;
  it_lab_ai::json model_data = it_lab_ai::read_json(json_file);

  if (comments) std::cout << "Loaded model data from JSON." << std::endl;

  auto input_layer = std::make_shared<it_lab_ai::InputLayer>(it_lab_ai::kNchw,
                                                             it_lab_ai::kNchw);
  input_layer->setName(it_lab_ai::kInput);
  layers.push_back(input_layer);
  name_to_layer["image_tensor"] = input_layer;
  int current_id = 0;
  input_layer->setID(current_id++); 
  for (const auto& layer_data : model_data) {
    try {
      std::string layer_name = layer_data["name"];
      int layer_index = layer_data["index"];
      std::string layer_type = layer_data["type"];

      if (layer_type == "InputLayer") continue;
      if (comments) {
        std::cout << "Processing layer " << layer_index << ": " << layer_name
                  << " (" << layer_type << ")" << std::endl;
      }

      std::shared_ptr<it_lab_ai::Layer> layer;

      if (layer_type.find("Conv") != std::string::npos) {
        it_lab_ai::Tensor tensor = it_lab_ai::create_tensor_from_json(
            layer_data, it_lab_ai::Type::kFloat);

        // Параметры по умолчанию
        size_t stride = 1;
        size_t pads = 0;
        size_t group = 1;
        std::vector<size_t> dilations = {1, 1};
        std::vector<size_t> pads_vec = {0, 0, 0,
                                        0};  // [top, bottom, left, right]

        // Извлекаем параметры из JSON
        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];

          if (attributes.contains("strides") &&
              attributes["strides"].is_array()) {
            auto strides = attributes["strides"];
            if (strides.size() >= 2) {
              stride = strides[0].get<size_t>();  // Используем первый stride
            }
          }

          if (attributes.contains("pads") && attributes["pads"].is_array()) {
            auto pads_array = attributes["pads"];
            if (pads_array.size() >= 4) {
              pads_vec = {
                  pads_array[0].get<size_t>(), pads_array[1].get<size_t>(),
                  pads_array[2].get<size_t>(), pads_array[3].get<size_t>()};
              // Используем симметричный padding (предполагаем, что top=bottom,
              // left=right)
              pads = pads_vec[0];
            }
          } else if (layer_data.contains("padding") &&
                     layer_data["padding"] == "valid") {
            pads = 0;
          } else if (layer_data.contains("padding") &&
                     layer_data["padding"] == "same") {
            // Для "same" padding вычисляем автоматически
            size_t kernel_size =
                tensor.get_shape()[0];  // предполагаем квадратное ядро
            pads = (kernel_size - 1) / 2;
          }

          if (attributes.contains("group")) {
            group = attributes["group"].get<size_t>();
          }

          if (attributes.contains("dilations") &&
              attributes["dilations"].is_array()) {
            auto dilations_array = attributes["dilations"];
            if (dilations_array.size() >= 2) {
              dilations = {dilations_array[0].get<size_t>(),
                           dilations_array[1].get<size_t>()};
            }
          }
        }

        // Транспонирование ядра (если нужно)
        it_lab_ai::Tensor tmp_tensor = tensor;
        /*
        // Раскомментируйте если нужно транспонирование
        for (size_t n = 0; n < tensor.get_shape()[2]; n++) {
            for (size_t c = 0; c < tensor.get_shape()[3]; c++) {
                for (size_t h = 0; h < tensor.get_shape()[0]; h++) {
                    for (size_t w = 0; w < tensor.get_shape()[1]; w++) {
                        tmp_tensor.set<float>({w, h, n, c},
                                            tensor.get<float>({h, w, n, c}));
                    }
                }
            }
        }
        */

        it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());

        // Создаем сверточный слой со всеми параметрами
        auto conv_layer = std::make_shared<it_lab_ai::ConvolutionalLayer>(
            stride, pads, group, tmp_tensor, tmp_bias, impl2);

        // Устанавливаем дополнительные параметры если они есть в реализации
        // (возможно нужно будет добавить методы setDilations, setPads в ваш
        // ConvolutionalLayer)
        conv_layer->setName(it_lab_ai::kConvolution);
        layer = conv_layer;
      } else if (layer_type.find("Relu") != std::string::npos ||
                 layer_type.find("relu") != std::string::npos) {
        auto ew_layer = std::make_shared<it_lab_ai::EWLayer>("relu");
        ew_layer->setName(it_lab_ai::kElementWise);
        layer = ew_layer;
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
      } else if (layer_type.find("MaxPool") != std::string::npos ||
                 layer_type.find("AveragePool") != std::string::npos) {
        std::string pooltype =
            (layer_type.find("Max") != std::string::npos) ? "max" : "average";

        // Параметры по умолчанию
        it_lab_ai::Shape shape = {2, 2};
        it_lab_ai::Shape strides = {2, 2};
        it_lab_ai::Shape pads = {0, 0, 0, 0};  // [top, bottom, left, right]
        it_lab_ai::Shape dilations = {1, 1};
        bool ceil_mode = false;

        // Извлекаем параметры из attributes
        if (layer_data.contains("attributes")) {
          const auto& attributes = layer_data["attributes"];

          // kernel_shape
          if (attributes.contains("kernel_shape") &&
              attributes["kernel_shape"].is_array()) {
            auto kernel_shape = attributes["kernel_shape"];
            if (kernel_shape.size() >= 2) {
              shape = it_lab_ai::Shape({kernel_shape[0].get<size_t>(),
                                        kernel_shape[1].get<size_t>()});
            }
          }

          // strides
          if (attributes.contains("strides") &&
              attributes["strides"].is_array()) {
            auto strides_array = attributes["strides"];
            if (strides_array.size() >= 2) {
              strides = it_lab_ai::Shape({strides_array[0].get<size_t>(),
                                          strides_array[1].get<size_t>()});
            }
          }

          // pads
          if (attributes.contains("pads") && attributes["pads"].is_array()) {
            auto pads_array = attributes["pads"];
            if (pads_array.size() >= 4) {
              pads = it_lab_ai::Shape(
                  {pads_array[0].get<size_t>(), pads_array[1].get<size_t>(),
                   pads_array[2].get<size_t>(), pads_array[3].get<size_t>()});
            }
          }

          // dilations
          if (attributes.contains("dilations") &&
              attributes["dilations"].is_array()) {
            auto dilations_array = attributes["dilations"];
            if (dilations_array.size() >= 2) {
              dilations = it_lab_ai::Shape({dilations_array[0].get<size_t>(),
                                            dilations_array[1].get<size_t>()});
            }
          }

          // ceil_mode
          if (attributes.contains("ceil_mode")) {
            ceil_mode = attributes["ceil_mode"].get<int>() != 0;
          }
        }

        // Создаем pooling слой
        auto pool_layer =
            std::make_shared<it_lab_ai::PoolingLayer>(shape, pooltype, impl1);

        // Устанавливаем дополнительные параметры, если они поддерживаются
        // (вам可能需要 добавить соответствующие методы в PoolingLayer)
        try {
          // Проверяем и устанавливаем strides
          if (strides[0] != 2 || strides[1] != 2) {
            pool_layer->setStrides(strides[0], strides[1]);
          }

          // Проверяем и устанавливаем padding
          if (pads[0] != 0 || pads[1] != 0 || pads[2] != 0 || pads[3] != 0) {
            pool_layer->setPads(pads[0], pads[1], pads[2], pads[3]);
          }

          // Проверяем и устанавливаем dilations
          if (dilations[0] != 1 || dilations[1] != 1) {
            pool_layer->setDilations(dilations[0], dilations[1]);
          }

          // Устанавливаем ceil_mode
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
        auto flatten_layer = std::make_shared<it_lab_ai::FlattenLayer>(
            std::vector<size_t>({0, 3, 2, 1}));
        flatten_layer->setName(it_lab_ai::kFlatten);
        layer = flatten_layer;
      } else if (layer_type == "Concat") {
        int axis = 0;
        if (layer_data.contains("axis")) {
          axis = layer_data["axis"];
        }
        auto concat_layer = std::make_shared<it_lab_ai::ConcatLayer>(axis);
        concat_layer->setName(it_lab_ai::kConcat);
        layer = concat_layer;
      } else if (layer_type == "Split") {
        int axis = 0;
        size_t num_outputs = 2;

        if (layer_data.contains("axis")) {
          axis = layer_data["axis"];
        }
        if (layer_data.contains("split") && layer_data["split"].is_array()) {
          num_outputs = layer_data["split"].size();
        }

        auto split_layer = std::make_shared<it_lab_ai::SplitLayer>(
            static_cast<int>(axis), static_cast<int>(num_outputs));
        split_layer->setName(it_lab_ai::kSplit);
        layer = split_layer;
      } else if (layer_type == "Add" || layer_type == "Mul" ||
                 layer_type == "Sub" || layer_type == "Div") {
        if (layer_data.contains("value")) {
          float value = 0.0f;
          if (layer_data["value"].is_string()) {
            try {
              value = std::stof(layer_data["value"].get<std::string>());
            } catch (...) {
              value = 0.0f;
            }
          } else if (layer_data["value"].is_number()) {
            value = layer_data["value"].get<float>();
          }

          std::string ew_operation;
          if (layer_type == "Mul") {
            ew_operation =
                "linear";
            auto ew_layer =
                std::make_shared<it_lab_ai::EWLayer>(ew_operation, value, 0.0f);
            ew_layer->setName(it_lab_ai::kElementWise);
            layer = ew_layer;
          } else if (layer_type == "Add") {
            ew_operation =
                "linear";
            auto ew_layer =
                std::make_shared<it_lab_ai::EWLayer>(ew_operation, 1.0f, value);
            ew_layer->setName(it_lab_ai::kElementWise);
            layer = ew_layer;
          } else if (layer_type == "Sub") {
            ew_operation =
                "linear";
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
        if (transB) {
          tmp_tensor = it_lab_ai::Tensor(
              it_lab_ai::Shape({tensor.get_shape()[1], tensor.get_shape()[0]}),
              it_lab_ai::Type::kFloat);

          for (size_t h = 0; h < tensor.get_shape()[0]; h++) {
            for (size_t w = 0; w < tensor.get_shape()[1]; w++) {
              tmp_tensor.set<float>({w, h}, tensor.get<float>({h, w}));
            }
          }
        }

        it_lab_ai::Tensor tmp_bias = it_lab_ai::make_tensor(tensor.get_bias());

        auto fc_layer =
            std::make_shared<it_lab_ai::FCLayer>(tmp_tensor, tmp_bias);
        fc_layer->setName(it_lab_ai::kFullyConnected);
        layer = fc_layer;
      } else {
        if (comments) {
          std::cout << "Warning: Unknown layer type: " << layer_type
                    << std::endl;
        }
        continue;
      }
      layer->setID(current_id++);
      layers.push_back(layer);
      name_to_layer[layer_name] = layer;
      if (layer_data.contains("inputs")) {
        for (const auto& input_name : layer_data["inputs"]) {
          std::string input_tensor = input_name.get<std::string>();
          connections[input_tensor].push_back(layer_name);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Error processing layer " << layer_data["index"] << " ("
                << layer_data["name"] << "): " << e.what() << std::endl;
      throw;
    }
  }

  if (comments) {
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
  }

  it_lab_ai::Graph graph(static_cast<int>(layers.size()));
  graph.setInput(*input_layer, input);

  if (comments) {
    std::cout << "\n=== CREATING GRAPH CONNECTIONS ===" << std::endl;
  }
  for (const auto& [source_tensor, target_layers] : connections) {
    std::string source_layer_name = get_base_layer_name(source_tensor);

    for (const auto& target_layer_name : target_layers) {
      connection_list.emplace_back(source_layer_name, target_layer_name);
    }
  }

  std::sort(connection_list.begin(), connection_list.end(),
            [&](const auto& a, const auto& b) {
              return name_to_layer[a.first]->getID() <
                     name_to_layer[b.first]->getID();
            });

  for (const auto& [source_name, target_name] : connection_list) {
    if (name_to_layer.count(source_name) && name_to_layer.count(target_name)) {
      try {
        graph.makeConnection(*name_to_layer[source_name],
                             *name_to_layer[target_name]);
      } catch (const std::exception& e) {
        std::cerr << "Failed: " << source_name << " -> " << target_name
                  << e.what()<<std::endl;
      }
    }
  }

  auto output_layer = layers.back();
  graph.setOutput(*output_layer, output);
  auto in_out_degrees = graph.getInOutDegrees();
  auto traversal_order = graph.getTraversalOrder();

  if (comments) {
    std::cout << "\n=== GRAPH TOPOLOGY ===" << std::endl;
    for (size_t i = 0; i < in_out_degrees.size(); i++) {
      std::cout << "Layer " << i << ": " << in_out_degrees[i].first
                << " inputs, " << in_out_degrees[i].second << " outputs"
                << std::endl;
    }

    std::cout << "Traversal order: ";
    for (int layer_id : traversal_order) {
      std::cout << layer_id << " ";
    }
    std::cout << std::endl;
  }
  
  if (comments) std::cout << "Starting inference..." << std::endl;
  try {
    graph.inference();
    if (comments) std::cout << "Inference completed successfully." << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "ERROR during inference: " << e.what() << std::endl;

    
  }

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