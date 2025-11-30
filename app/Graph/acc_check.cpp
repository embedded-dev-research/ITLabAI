#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "build.cpp"
#include "build.hpp"

namespace fs = std::filesystem;
using namespace it_lab_ai;

int main(int argc, char* argv[]) {
  std::string model_name = "alexnet_mnist";
  bool onednn = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (std::string(argv[i]) == "--onednn") {
      onednn = true;
    }
  }
  it_lab_ai::LayerFactory::configure(onednn);
  std::string dataset_path;
  if (model_name == "alexnet_mnist") {
    dataset_path = MNIST_PATH;
  } else {
    dataset_path = IMAGENET_ACC;
  }

  std::string json_path = model_paths[model_name];
  std::vector<int> input_shape = get_input_shape_from_json(json_path);

  std::cout << std::endl;

  if (model_name == "alexnet_mnist") {
    std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                  891, 957,  1027, 973,  1008};
    int stat = 0;
    size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
    int count_pic = static_cast<int>(sum) + 10;
    std::vector<float> res(count_pic * 28 * 28);
    Tensor input;
    Shape sh1({1, 5, 5, 3});
    std::vector<float> vec;
    vec.reserve(75);
    for (int i = 0; i < 75; ++i) {
      vec.push_back(3);
    }
    Tensor output = make_tensor(vec, sh1);

    for (size_t name = 0; name < 10; name++) {
      for (size_t ind = 0; ind < counts[name] + 1; ind++) {
        std::ostringstream oss;
        oss << "/" << name << "_" << std::setw(6) << std::setfill('0') << ind
            << ".png";
        std::string png = oss.str();
        std::string image_path = MNIST_PATH + png;

        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
          throw std::runtime_error("Failed to load image");
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        std::vector<cv::Mat> channels;
        cv::split(image, channels);
        for (int i = 0; i < 28; ++i) {
          for (int j = 0; j < 28; ++j) {
            size_t a = ind;
            for (size_t n = 0; n < name; n++) a += counts[n] + 1;
            res[(a) * 28 * 28 + i * 28 + j] = channels[0].at<uchar>(j, i);
          }
        }
      }
    }
    Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
    Tensor t = make_tensor<float>(res, sh);
    input = t;
    auto graph = build_graph_linear(input, output, false);
    graph.inference();
    print_time_stats(graph);
    std::vector<std::vector<float>> tmp_output =
        softmax<float>(*output.as<float>(), 10);
    std::vector<size_t> indices;
    for (const auto& row : tmp_output) {
      for (size_t j = 0; j < row.size(); ++j) {
        if (row[j] >= 1e-6) {
          indices.push_back(j);
          break;
        }
      }
    }
    for (size_t name = 0; name < 10; name++) {
      for (size_t ind = 0; ind < counts[name] + 1; ind++) {
        size_t a = ind;
        for (size_t n = 0; n < name; n++) a += counts[n] + 1;
        if (name == indices[a]) stat++;
      }
    }
    double percentage =
        (static_cast<double>(stat) / static_cast<double>(sum + 10)) * 100;
    std::cout << "Stat: " << std::fixed << std::setprecision(2) << percentage
              << "%" << std::endl;
    return 0;
  }
  std::vector<size_t> counts;
  std::vector<std::string> image_paths;
  std::vector<int> true_labels;
  std::vector<float> all_image_data;
  size_t total_images = 0;

  counts.resize(1000, 0);

  for (int class_id = 0; class_id < 1000; ++class_id) {
    std::ostringstream folder_oss;
    folder_oss << std::setw(5) << std::setfill('0') << class_id;
    std::string class_folder_path = dataset_path + "/" + folder_oss.str();

    if (fs::exists(class_folder_path)) {
      for (const auto& entry : fs::directory_iterator(class_folder_path)) {
        if (entry.path().extension() == ".png" ||
            entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg") {
          counts[class_id]++;
          total_images++;
        }
      }
    }
  }

  if (total_images == 0) {
    std::cerr << "No images found in dataset path: " << dataset_path
              << std::endl;
    return 1;
  }

  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  size_t image_size = channels * height * width;

  all_image_data.resize(total_images * image_size);

  size_t current_index = 0;
  for (int class_id = 0; class_id < 1000; ++class_id) {
    std::ostringstream folder_oss;
    folder_oss << std::setw(5) << std::setfill('0') << class_id;
    std::string class_folder_path = dataset_path + "/" + folder_oss.str();

    if (!fs::exists(class_folder_path)) continue;

    for (const auto& entry : fs::directory_iterator(class_folder_path)) {
      if (entry.path().extension() == ".png" ||
          entry.path().extension() == ".jpg" ||
          entry.path().extension() == ".jpeg") {
        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty()) {
          std::cerr << "Failed to load image: " << entry.path().string()
                    << std::endl;
          continue;
        }

        it_lab_ai::Tensor prepared_tensor =
            prepare_image(image, input_shape, model_name);
        const std::vector<float>& image_data = *prepared_tensor.as<float>();

        std::copy(image_data.begin(), image_data.end(),
                  all_image_data.begin() + current_index * image_size);

        image_paths.push_back(entry.path().string());
        true_labels.push_back(class_id);
        current_index++;
      }
    }
  }

  it_lab_ai::Shape input_shape_imagenet(
      {total_images, static_cast<size_t>(channels), static_cast<size_t>(height),
       static_cast<size_t>(width)});
  it_lab_ai::Tensor input =
      it_lab_ai::make_tensor(all_image_data, input_shape_imagenet);

  size_t output_classes = 1000;
  it_lab_ai::Shape output_shape({total_images, output_classes});
  it_lab_ai::Tensor output =
      it_lab_ai::Tensor(output_shape, it_lab_ai::Type::kFloat);

  auto graph = build_graph(input, output, json_path, false);
  graph.inference();
  print_time_stats(graph);
  std::vector<std::vector<float>> processed_outputs;
  const std::vector<float>& raw_output = *output.as<float>();

  for (size_t i = 0; i < total_images; ++i) {
    std::vector<float> single_output(
        raw_output.begin() + i * output_classes,
        raw_output.begin() + (i + 1) * output_classes);
    std::vector<float> processed_output =
        process_model_output(single_output, model_name);
    processed_outputs.push_back(processed_output);
  }

  int correct_predictions_top1 = 0;
  int correct_predictions_top5 = 0;
  for (size_t i = 0; i < processed_outputs.size(); ++i) {
    int true_label = true_labels[i];
    const std::vector<float>& probabilities = processed_outputs[i];

    std::vector<size_t> indices(probabilities.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
      return probabilities[a] > probabilities[b];
    });

    size_t predicted_class_top1 = indices[0];
    if (predicted_class_top1 == static_cast<size_t>(true_label)) {
      correct_predictions_top1++;
    }

    bool found_in_top5 = false;
    for (int top_k = 0; top_k < std::min(5, static_cast<int>(indices.size()));
         ++top_k) {
      if (indices[top_k] == static_cast<size_t>(true_label)) {
        found_in_top5 = true;
        break;
      }
    }
    if (found_in_top5) {
      correct_predictions_top5++;
    }
  }

  double final_accuracy_top1 =
      (static_cast<double>(correct_predictions_top1) / total_images) * 100;
  double final_accuracy_top5 =
      (static_cast<double>(correct_predictions_top5) / total_images) * 100;

  std::cout << "\nFinal Results:" << std::endl;
  std::cout << "Model: " << model_name << std::endl;
  std::cout << "Dataset: " << dataset_path << std::endl;
  std::cout << "Total images: " << total_images << std::endl;
  std::cout << "Correct predictions (Top-1): " << correct_predictions_top1
            << std::endl;
  std::cout << "Correct predictions (Top-5): " << correct_predictions_top5
            << std::endl;
  std::cout << "Top-1 Accuracy: " << std::fixed << std::setprecision(2)
            << final_accuracy_top1 << "%" << std::endl;
  std::cout << "Top-5 Accuracy: " << std::fixed << std::setprecision(2)
            << final_accuracy_top5 << "%" << std::endl;

  return 0;
}