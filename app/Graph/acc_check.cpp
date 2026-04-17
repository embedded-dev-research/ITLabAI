#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "build.hpp"

namespace fs = std::filesystem;
using namespace it_lab_ai;

int main(int argc, char* argv[]) {
  std::string model_name = "alexnet_mnist";
  RuntimeOptions options;
  size_t num_photo = 1000;
  size_t batch_size = 250;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (std::string(argv[i]) == "--onednn") {
      options.backend = Backend::kOneDnn;
      if (options.par_backend != ParBackend::kSeq) {
        std::cout << "Warning: oneDNN backend is not compatible with parallel "
                     "execution. Disabling parallelism."
                  << '\n';
        options.par_backend = ParBackend::kSeq;
      }
    } else if (std::string(argv[i]) == "--parallel" && i + 1 < argc) {
      if (options.backend == Backend::kOneDnn) {
        std::cout << "Warning: Parallel execution is not compatible with "
                     "oneDNN backend. Ignoring --parallel option."
                  << '\n';
        i++;
        continue;
      }

      std::string backend_str = argv[++i];
      if (backend_str == "tbb") {
        options.par_backend = ParBackend::kTbb;
      } else if (backend_str == "threads" || backend_str == "stl") {
        options.par_backend = ParBackend::kThreads;
      } else if (backend_str == "omp") {
        options.par_backend = ParBackend::kOmp;
      } else {
        std::cerr << "Unknown parallel backend: " << backend_str
                  << ". Using default (Threads)." << '\n';
        options.par_backend = ParBackend::kThreads;
      }
    } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
      options.threads = std::stoi(argv[++i]);
    } else {
      try {
        size_t input_value = std::stoul(argv[i]);

        if (input_value < 1) {
          std::cerr << "Warning: num_photo cannot be less than 1. Using 1."
                    << '\n';
          num_photo = 1;
        } else if (input_value > 50000) {
          std::cerr << "Warning: num_photo cannot exceed 50000. Using 50000."
                    << '\n';
          num_photo = 50000;
        } else {
          num_photo = input_value;
        }
      } catch (const std::exception& e) {
        std::cerr << "Error: Invalid numeric argument: " << e.what() << argv[i]
                  << ". Using default value: 1000" << '\n';
        num_photo = 1000;
      }
    }
  }

  std::string dataset_path;
  if (model_name == "alexnet_mnist") {
    dataset_path = MNIST_PATH;
  } else {
    dataset_path = IMAGENET_ACC;
  }

  std::string json_path = model_paths[model_name];
  std::vector<int> input_shape = get_input_shape_from_json(json_path);

  std::cout << '\n';

  if (model_name == "alexnet_mnist") {
    std::vector<size_t> counts = {979, 1134, 1031, 1009, 981,
                                  891, 957,  1027, 973,  1008};
    int stat = 0;
    size_t sum = std::accumulate(counts.begin(), counts.end(), size_t{0});
    int count_pic = static_cast<int>(sum) + 10;
    std::vector<float> res(count_pic * 28 * 28);
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
            for (size_t n = 0; n < name; n++) {
              a += counts[n] + 1;
            }
            res[(a) * 28 * 28 + i * 28 + j] = channels[0].at<uchar>(j, i);
          }
        }
      }
    }
    Shape sh({static_cast<size_t>(count_pic), 1, 28, 28});
    Tensor t = make_tensor<float>(res, sh);
    Tensor input = t;
    Graph graph;
    build_graph_linear(graph, input, output, options, false);
    graph.inference(options);
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
        for (size_t n = 0; n < name; n++) {
          a += counts[n] + 1;
        }
        if (name == indices[a]) {
          stat++;
        }
      }
    }
    double percentage =
        (static_cast<double>(stat) / static_cast<double>(sum + 10)) * 100;
    std::cout << "Stat: " << std::fixed << std::setprecision(2) << percentage
              << "%" << '\n';

    return 0;
  }

  size_t output_classes = 1000;
  std::vector<size_t> counts(output_classes, 0);

  for (size_t class_id = 0; class_id < output_classes; ++class_id) {
    std::ostringstream folder_oss;
    folder_oss << std::setw(5) << std::setfill('0') << class_id;
    std::string class_folder_path = dataset_path + "/" + folder_oss.str();

    if (fs::exists(class_folder_path)) {
      for (const auto& entry : fs::directory_iterator(class_folder_path)) {
        if (entry.path().extension() == ".png" ||
            entry.path().extension() == ".jpg" ||
            entry.path().extension() == ".jpeg") {
          counts[class_id]++;
        }
      }
    }
  }

  size_t images_per_class_base = num_photo / output_classes;
  size_t remaining = num_photo % output_classes;

  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  size_t image_size = channels * height * width;

  int correct_predictions_top1 = 0;
  int correct_predictions_top5 = 0;

  Graph graph;
  std::shared_ptr<Layer> input_layer = nullptr;
  std::shared_ptr<Layer> output_layer = nullptr;
  bool graph_built = false;

  int total_inference_time = 0;
  int batch_count = 0;
  size_t total_processed = 0;

  std::vector<float> batch_data;
  std::vector<int> batch_labels;
  batch_data.reserve(batch_size * image_size);
  batch_labels.reserve(batch_size);

  for (size_t class_id = 0;
       class_id < output_classes && total_processed < num_photo; ++class_id) {
    size_t need_from_class = images_per_class_base;
    if (remaining > 0) {
      need_from_class++;
      remaining--;
    }

    if (need_from_class == 0) {
      continue;
    }

    std::ostringstream folder_oss;
    folder_oss << std::setw(5) << std::setfill('0') << class_id;
    std::string class_folder_path = dataset_path + "/" + folder_oss.str();

    if (!fs::exists(class_folder_path)) {
      continue;
    }

    std::vector<std::string> class_images;
    for (const auto& entry : fs::directory_iterator(class_folder_path)) {
      if (entry.path().extension() == ".jpg" ||
          entry.path().extension() == ".jpeg" ||
          entry.path().extension() == ".png") {
        class_images.push_back(entry.path().string());
      }
    }

    size_t images_to_take = std::min(need_from_class, class_images.size());

    for (size_t img_idx = 0; img_idx < images_to_take; ++img_idx) {
      cv::Mat image = cv::imread(class_images[img_idx]);
      if (image.empty()) {
        std::cerr << "Failed to load image: " << class_images[img_idx] << '\n';
        continue;
      }

      it_lab_ai::Tensor prepared =
          prepare_image(image, input_shape, model_name);
      const auto& img_data = *prepared.as<float>();

      batch_data.insert(batch_data.end(), img_data.begin(), img_data.end());
      batch_labels.push_back(static_cast<int>(class_id));
      total_processed++;

      if (batch_labels.size() >= batch_size) {
        it_lab_ai::Shape batch_input_shape(
            {batch_labels.size(), static_cast<size_t>(channels),
             static_cast<size_t>(height), static_cast<size_t>(width)});
        it_lab_ai::Tensor batch_input =
            make_tensor(batch_data, batch_input_shape);

        it_lab_ai::Shape batch_output_shape(
            {batch_labels.size(), output_classes});
        it_lab_ai::Tensor batch_output(batch_output_shape,
                                       it_lab_ai::Type::kFloat);

        if (!graph_built) {
          build_graph(graph, batch_input, batch_output, json_path, options,
                      false);

          for (int i = 0; i < graph.getLayersCount(); ++i) {
            auto layer = graph.getLayerFromID(i);
            if (layer->getName() == kInput) {
              input_layer = layer;
            }
            if (i == graph.getLayersCount() - 1) {
              output_layer = layer;
            }
          }

          if (!input_layer || !output_layer) {
            std::cerr << "Error: Could not find input/output layers" << '\n';
            return 1;
          }

          graph_built = true;
        } else {
          graph.setInput(input_layer, batch_input);
          graph.setOutput(output_layer, batch_output);
        }

        auto batch_start = std::chrono::high_resolution_clock::now();
        graph.inference(options);
        auto batch_end = std::chrono::high_resolution_clock::now();
        int batch_time = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(batch_end -
                                                                  batch_start)
                .count());

        total_inference_time += batch_time;
        batch_count++;

        const std::vector<float>& raw_batch_output = *batch_output.as<float>();
        const float* data_ptr = raw_batch_output.data();

        for (size_t i = 0; i < batch_labels.size(); ++i) {
          const float* img_start = data_ptr + i * output_classes;
          int true_label = batch_labels[i];

          std::vector<std::pair<float, int>> top5;
          for (size_t j = 0; j < output_classes; ++j) {
            float val = img_start[j];

            if (top5.size() < 5) {
              top5.emplace_back(val, static_cast<int>(j));
              std::sort(top5.begin(), top5.end(),
                        [](auto& a, auto& b) { return a.first > b.first; });
            } else if (val > top5.back().first) {
              top5.back() = {val, static_cast<int>(j)};
              std::sort(top5.begin(), top5.end(),
                        [](auto& a, auto& b) { return a.first > b.first; });
            }
          }

          if (top5[0].second == true_label) {
            correct_predictions_top1++;
          }
          for (int k = 0; k < 5; ++k) {
            if (top5[k].second == true_label) {
              correct_predictions_top5++;
              break;
            }
          }
        }

        batch_data.clear();
        batch_labels.clear();
        batch_data.reserve(batch_size * image_size);
        batch_labels.reserve(batch_size);
      }
    }
  }

  if (!batch_data.empty()) {
    it_lab_ai::Shape batch_input_shape(
        {batch_labels.size(), static_cast<size_t>(channels),
         static_cast<size_t>(height), static_cast<size_t>(width)});
    it_lab_ai::Tensor batch_input = make_tensor(batch_data, batch_input_shape);

    it_lab_ai::Shape batch_output_shape({batch_labels.size(), output_classes});
    it_lab_ai::Tensor batch_output(batch_output_shape, it_lab_ai::Type::kFloat);

    if (!graph_built) {
      build_graph(graph, batch_input, batch_output, json_path, options, false);

      for (int i = 0; i < graph.getLayersCount(); ++i) {
        auto layer = graph.getLayerFromID(i);
        if (layer->getName() == kInput) {
          input_layer = layer;
        }
        if (i == graph.getLayersCount() - 1) {
          output_layer = layer;
        }
      }
    } else {
      graph.setInput(input_layer, batch_input);
      graph.setOutput(output_layer, batch_output);
    }

    auto batch_start = std::chrono::high_resolution_clock::now();
    graph.inference(options);
    auto batch_end = std::chrono::high_resolution_clock::now();
    int batch_time =
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             batch_end - batch_start)
                             .count());

    total_inference_time += batch_time;
    batch_count++;

    const std::vector<float>& raw_batch_output = *batch_output.as<float>();
    const float* data_ptr = raw_batch_output.data();

    for (size_t i = 0; i < batch_labels.size(); ++i) {
      const float* img_start = data_ptr + i * output_classes;
      int true_label = batch_labels[i];

      std::vector<std::pair<float, int>> top5;
      for (size_t j = 0; j < output_classes; ++j) {
        float val = img_start[j];
        if (top5.size() < 5) {
          top5.emplace_back(val, static_cast<int>(j));
          std::sort(top5.begin(), top5.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
        } else if (val > top5.back().first) {
          top5.back() = {val, static_cast<int>(j)};
          std::sort(top5.begin(), top5.end(),
                    [](auto& a, auto& b) { return a.first > b.first; });
        }
      }

      if (top5[0].second == true_label) {
        correct_predictions_top1++;
      }
      for (int k = 0; k < 5; ++k) {
        if (top5[k].second == true_label) {
          correct_predictions_top5++;
          break;
        }
      }
    }
  }

  std::cout << "Total inference time: " << total_inference_time << " ms\n";
  std::cout << "Number of batches: " << batch_count << '\n';
  if (batch_count > 0) {
    std::cout << "Average time per batch: "
              << total_inference_time / batch_count << " ms\n";
  }

  double final_accuracy_top1 =
      (static_cast<double>(correct_predictions_top1) / num_photo) * 100;
  double final_accuracy_top5 =
      (static_cast<double>(correct_predictions_top5) / num_photo) * 100;

  std::cout << "\nFinal Results:" << '\n';
  std::cout << "Model: " << model_name << '\n';
  std::cout << "Dataset: " << dataset_path << '\n';
  std::cout << "Total images: " << num_photo << '\n';
  std::cout << "Batch size: " << batch_size << '\n';
  std::cout << "Correct predictions (Top-1): " << correct_predictions_top1
            << '\n';
  std::cout << "Correct predictions (Top-5): " << correct_predictions_top5
            << '\n';
  std::cout << "Top-1 Accuracy: " << std::fixed << std::setprecision(2)
            << final_accuracy_top1 << "%" << '\n';
  std::cout << "Top-5 Accuracy: " << std::fixed << std::setprecision(2)
            << final_accuracy_top5 << "%" << '\n';

  return 0;
}