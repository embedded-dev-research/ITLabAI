#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "build.cpp"
#include "build.hpp"

namespace fs = std::filesystem;
using namespace it_lab_ai;

std::unordered_map<int, std::string> load_class_names(
    const std::string& filename) {
  std::unordered_map<int, std::string> class_names;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    throw std::runtime_error("Cannot open class names file: " + filename);
  }

  while (std::getline(file, line)) {
    line = std::regex_replace(line, std::regex("^\\s+|\\s+$"), "");
    if (line.empty()) continue;

    std::regex pattern("(\\d+):\\s*'([^']+)'");
    std::smatch matches;

    if (std::regex_search(line, matches, pattern)) {
      int class_id = std::stoi(matches[1]);
      std::string class_name = matches[2];
      class_names[class_id] = class_name;
    }
  }

  return class_names;
}

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
        auto shape = attributes["shape"].get<std::vector<int>>();

        if (shape.size() == 2) {
          if (shape[1] == 784) {
            return {shape[0], 1, 28, 28};
          }
        } else if (shape.size() == 4) {
          return shape;
        }
      }
    }
  }
  return {28};
}

std::vector<float> process_model_output(const std::vector<float>& output,
                                        const std::string& model_name) {
  bool is_yolo = (model_name.find("yolo") != std::string::npos);

  if (!is_yolo) {
    return softmax<float>(output);
  }
  float sum_val = std::accumulate(output.begin(), output.end(), 0.0f);
  if (std::abs(sum_val - 1.0f) < 0.01f) {
    std::cout << "YOLO output already normalized, using as-is" << std::endl;
    return output;
  }
  std::cout << "Applying softmax to YOLO output" << std::endl;
  return softmax<float>(output);
}

it_lab_ai::Tensor prepare_image(const cv::Mat& image,
                                const std::vector<int>& input_shape,
                                const std::string& model_name = "") {
  if (input_shape.size() != 4) {
    throw std::runtime_error("Input shape must have 4 dimensions");
  }

  int batch_size = input_shape[0];
  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];

  cv::Mat processed_image;
  cv::Size target_size(width, height);

  bool is_yolo_model = (model_name.find("yolo") != std::string::npos ||
                        model_name.find("Google"));

  if (image.rows == height && image.cols == width) {
    processed_image = image.clone();
    std::cout << "Image already at target size - no resize needed" << std::endl;
  } else {
    if (is_yolo_model) {
      double scale = std::min(static_cast<double>(width) / image.cols,
                              static_cast<double>(height) / image.rows);
      int new_width = static_cast<int>(image.cols * scale);
      int new_height = static_cast<int>(image.rows * scale);

      cv::Mat resized_image;
      cv::resize(image, resized_image, cv::Size(new_width, new_height), 0, 0,
                 cv::INTER_LINEAR);

      processed_image = cv::Mat::zeros(height, width, image.type());
      int x_offset = (width - new_width) / 2;
      int y_offset = (height - new_height) / 2;
      resized_image.copyTo(
          processed_image(cv::Rect(x_offset, y_offset, new_width, new_height)));

      std::cout << "YOLO resize with padding applied" << std::endl;
    } else {
      int interpolation = cv::INTER_LINEAR;
      if (image.rows < height || image.cols < width) {
        interpolation = cv::INTER_CUBIC;
      } else if (image.rows > height * 2 || image.cols > width * 2) {
        interpolation = cv::INTER_AREA;
      }
      cv::resize(image, processed_image, target_size, 0, 0, interpolation);
      std::cout << "Standard resize applied" << std::endl;
    }
  }

  cv::Mat float_image;
  processed_image.convertTo(float_image, CV_32FC3);

  if (is_yolo_model) {
    float_image /= 255.0;
    std::cout << "YOLO normalization: 0-1 range" << std::endl;
  } else {
    float_image /= 255.0;
    if (channels == 3) {
      std::vector<cv::Mat> image_channels;
      cv::split(float_image, image_channels);

      image_channels[0] = (image_channels[0] - 0.485) / 0.229;
      image_channels[1] = (image_channels[1] - 0.456) / 0.224;
      image_channels[2] = (image_channels[2] - 0.406) / 0.225;

      cv::merge(image_channels, float_image);
      std::cout << "ImageNet normalization applied" << std::endl;
    } else if (channels == 1) {
      cv::cvtColor(float_image, float_image, cv::COLOR_BGR2GRAY);
    }
  }

  std::vector<float> data;
  data.reserve(batch_size * channels * height * width);
  std::vector<cv::Mat> processed_channels;
  cv::split(float_image, processed_channels);
  if (!is_yolo_model && channels == 3) {
    std::swap(processed_channels[0], processed_channels[2]);
  }

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

it_lab_ai::Tensor prepare_mnist_image(const cv::Mat& image) {
  cv::Mat gray_image;
  cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
  std::vector<cv::Mat> channels;
  cv::split(image, channels);

  std::vector<float> res(28 * 28);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      res[i * 28 + j] = channels[0].at<uchar>(j, i);
    }
  }

  Shape sh({1, 1, 28, 28});
  return it_lab_ai::make_tensor(res, sh);
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
  input_shape = get_input_shape_from_json(json_path);

  std::string image_folder;
  if (model_name == "alexnet_mnist") {
    image_folder = IMAGE28_PATH;
  } else {
    image_folder = IMAGENET_PATH;
  }
  std::cout << "Using image folder: " << image_folder << std::endl;

  std::vector<std::string> image_paths;
  for (const auto& entry : fs::directory_iterator(image_folder)) {
    if (entry.path().extension() == ".png" ||
        entry.path().extension() == ".jpg" ||
        entry.path().extension() == ".jpeg") {
      image_paths.push_back(entry.path().string());
    }
  }

  std::cout << "Found " << image_paths.size() << " images to process"
            << std::endl;

  std::unordered_map<int, std::string> class_names;
  try {
    class_names = load_class_names(IMAGENET_LABELS);
  } catch (const std::exception& e) {
    std::cerr << "Warning: " << e.what() << std::endl;
  }

  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Failed to load image: " << image_path << std::endl;
      continue;
    }

    try {
      std::cout << "\nProcessing image: " << image_path << std::endl;
      std::cout << "Original size: " << image.cols << "x" << image.rows
                << ", channels: " << image.channels() << std::endl;

      if (model_name == "alexnet_mnist") {
        it_lab_ai::Tensor input = prepare_mnist_image(image);
        it_lab_ai::Shape sh1({1, 5, 5, 3});
        std::vector<float> vec(75, 3);
        it_lab_ai::Tensor output = it_lab_ai::make_tensor(vec, sh1);

        build_graph_linear(input, output, true, parallel);
        std::vector<float> tmp_output = softmax<float>(*output.as<float>());
        int top_n = std::min(3, static_cast<int>(tmp_output.size()));
        std::vector<int> indices(tmp_output.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + top_n, indices.end(),
            [&](int a, int b) { return tmp_output[a] > tmp_output[b]; });

        std::cout << "Top " << top_n << " predictions for MNIST:" << std::endl;
        for (int i = 0; i < top_n; i++) {
          int idx = indices[i];
          std::cout << "  " << (i + 1) << ". Class " << idx << ": "
                    << std::fixed << std::setprecision(6)
                    << tmp_output[idx] * 100 << "%" << std::endl;
        }

        int max_class = indices[0];
        float max_prob = tmp_output[max_class];
        std::cout << "Image: " << fs::path(image_path).filename().string()
                  << " -> Predicted digit: " << max_class
                  << " (probability: " << std::fixed << std::setprecision(6)
                  << max_prob * 100 << "%)" << std::endl;

      } else {
        it_lab_ai::Tensor input = prepare_image(image, input_shape, model_name);

        size_t output_classes = 1000;
        it_lab_ai::Tensor output({1, output_classes}, it_lab_ai::Type::kFloat);

        build_graph(input, output, json_path, false, parallel);
        std::vector<float> tmp_output =
            process_model_output(*output.as<float>(), model_name);

        int top_n = std::min(5, static_cast<int>(tmp_output.size()));
        std::vector<int> indices(tmp_output.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + top_n, indices.end(),
            [&](int a, int b) { return tmp_output[a] > tmp_output[b]; });

        std::cout << "Top " << top_n << " predictions:" << std::endl;
        for (int i = 0; i < top_n; i++) {
          int idx = indices[i];
          std::cout << "  " << (i + 1) << ". Class " << idx << ": "
                    << std::fixed << std::setprecision(6) << tmp_output[idx];

          if (class_names.find(idx) != class_names.end()) {
            std::cout << " (" << class_names[idx] << ")";
          }
          std::cout << std::endl;
        }

        int max_class = indices[0];
        float max_prob = tmp_output[max_class];
        std::cout << "Image: " << fs::path(image_path).filename().string()
                  << " -> Predicted class: " << max_class;
        if (class_names.find(max_class) != class_names.end()) {
          std::cout << " (" << class_names[max_class] << ")";
        }
        std::cout << " (probability: " << std::fixed << std::setprecision(6)
                  << max_prob << ")" << std::endl;
      }
      std::cout << "----------------------------------------" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error processing image " << image_path << ": " << e.what()
                << std::endl;
    }
  }
  return 0;
}