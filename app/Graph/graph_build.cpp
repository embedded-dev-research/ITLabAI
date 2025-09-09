#include "build.cpp"
#include "build.hpp"
#include <unordered_map>

namespace fs = std::filesystem;
using namespace it_lab_ai;

std::unordered_map<std::string, std::string> model_paths = {
    {"alexnet_mnist", MODEL_PATH_H5},
    {"googlenet", MODEL_PATH_GOOGLENET_ONNX},
    {"resnet", MODEL_PATH_RESNET_ONNX},
    {"densenet", MODEL_PATH_DENSENET_ONNX},
    {"yolo", MODEL_PATH_YOLO11NET_ONNX}};

std::vector<int> get_input_shape_from_json(const std::string& json_path) {
  it_lab_ai::json model_data = it_lab_ai::read_json(json_path);

  for (const auto& layer_data : model_data) {
    if (layer_data["type"] == "InputLayer" &&
        layer_data.contains("attributes")) {
      auto attributes = layer_data["attributes"];
      if (attributes.contains("shape")) {
        return attributes["shape"].get<std::vector<int>>();
      }
    }
  }

  throw std::runtime_error("Could not determine input shape from JSON");
}

it_lab_ai::Tensor prepare_image(const cv::Mat& image,
                                const std::vector<int>& input_shape) {
  if (input_shape.size() != 4) {
    throw std::runtime_error("Input shape must have 4 dimensions");
  }

  int batch_size = input_shape[0];
  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(width, height));

  cv::Mat float_image;
  resized.convertTo(float_image, CV_32FC3);
  float_image /= 255.0;

  if (channels == 3) {
    std::vector<cv::Mat> image_channels;
    cv::split(float_image, image_channels);

    image_channels[0] = (image_channels[0] - 0.485) / 0.229;
    image_channels[1] = (image_channels[1] - 0.456) / 0.224;
    image_channels[2] = (image_channels[2] - 0.406) / 0.225;

    cv::merge(image_channels, float_image);
  }

  else if (channels == 1) {
    cv::cvtColor(float_image, float_image, cv::COLOR_BGR2GRAY);
  }

  std::vector<float> data;
  data.reserve(batch_size * channels * height * width);


  std::vector<cv::Mat> processed_channels;
  cv::split(float_image, processed_channels);

  for (int c = 0; c < channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data.push_back(processed_channels[c].at<float>(h, w));
      }
    }
  }


  it_lab_ai::Shape shape(
      {static_cast<size_t>(batch_size), static_cast<size_t>(channels),
       static_cast<size_t>(height), static_cast<size_t>(width)});

  return it_lab_ai::make_tensor(data, shape);
}

int main(int argc, char* argv[]) {
  std::string model_name = "alexnet_mnist";
  bool parallel = false;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--parallel") {
      parallel = true;
    } else if (std::string(argv[i]) == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    }
  }

  std::string json_path = model_paths[model_name];

  std::vector<int> input_shape;
  try {
    input_shape = get_input_shape_from_json(json_path);
    std::cout << "Input shape from JSON: [";
    for (size_t i = 0; i < input_shape.size(); ++i) {
      std::cout << input_shape[i];
      if (i < input_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error reading input shape: " << e.what() << std::endl;
    return 1;
  }

  std::string image_folder;
  if (input_shape[2] == 28 && input_shape[3] == 28) {
    image_folder = IMAGE28_PATH;
  } else if (input_shape[2] == 224 && input_shape[3] == 224) {
    image_folder = IMAGE224_PATH;
  } else if (input_shape[2] == 256 && input_shape[3] == 256) {
    image_folder = IMAGE256_PATH;
  } else {
    image_folder = IMAGE28_PATH;
  }
  std::vector<std::string> image_paths;

  for (const auto& entry : fs::directory_iterator(image_folder)) {
    if (entry.path().extension() == ".png" ||
        entry.path().extension() == ".jpg") {
      image_paths.push_back(entry.path().string());
    }
  }

  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) continue;

    try {
      it_lab_ai::Tensor input = prepare_image(image, input_shape);

      it_lab_ai::Tensor output({1, 1000}, it_lab_ai::Type::kFloat);

      build_graph(input, output, json_path, true, parallel);

      std::vector<float> tmp_output = softmax<float>(*output.as<float>());
      for (size_t i = 0; i < tmp_output.size(); i++) {
        if (tmp_output[i] >= 1e-6) {
          std::cout << "Image: " << image_path << " -> Class: " << i
                    << std::endl;
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "Error processing image " << image_path << ": " << e.what()
                << std::endl;
    }
  }
  return 0;
}