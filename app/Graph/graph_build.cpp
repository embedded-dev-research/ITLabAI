#include <algorithm>
#include <numeric>
#include <unordered_map>

#include "build.hpp"

namespace fs = std::filesystem;
using namespace it_lab_ai;

int main(int argc, char* argv[]) {
  std::string model_name = "alexnet_mnist";
  RuntimeOptions options;

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
      } else if (backend_str == "kokkos") {
        options.par_backend = ParBackend::kKokkos;
      } else {
        std::cerr << "Unknown parallel backend: " << backend_str
                  << ". Using default (Threads)." << '\n';
        options.par_backend = ParBackend::kThreads;
      }
    } else if (std::string(argv[i]) == "--threads" && i + 1 < argc) {
      options.threads = std::stoi(argv[++i]);
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

  std::vector<std::string> image_paths;
  for (const auto& entry : fs::directory_iterator(image_folder)) {
    if (entry.path().extension() == ".png" ||
        entry.path().extension() == ".jpg" ||
        entry.path().extension() == ".jpeg") {
      image_paths.push_back(entry.path().string());
    }
  }

  std::unordered_map<int, std::string> class_names;
  try {
    class_names = load_class_names(IMAGENET_LABELS);
  } catch (const std::exception& e) {
    std::cerr << "Warning: " << e.what() << '\n';
  }

  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      std::cerr << "Failed to load image: " << image_path << '\n';
      continue;
    }

    try {
      if (model_name == "alexnet_mnist") {
        it_lab_ai::Tensor input = prepare_mnist_image(image);
        it_lab_ai::Shape sh1({1, 5, 5, 3});
        std::vector<float> vec(75, 3);
        it_lab_ai::Tensor output = it_lab_ai::make_tensor(vec, sh1);
        Graph graph;
        build_graph_linear(graph, input, output, options, true);

        std::cout << "Starting inference..." << '\n';
        try {
          graph.inference(options);
          std::cout << "Inference completed successfully." << '\n';
        } catch (const std::exception& e) {
          std::cerr << "ERROR during inference: " << e.what() << '\n';
        }
        print_time_stats(graph);
        std::vector<float> tmp_output = softmax<float>(*output.as<float>());
        int top_n = std::min(3, static_cast<int>(tmp_output.size()));
        std::vector<int> indices(tmp_output.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + top_n, indices.end(),
            [&](int a, int b) { return tmp_output[a] > tmp_output[b]; });

        std::cout << "Top " << top_n << " predictions for MNIST:" << '\n';
        for (int i = 0; i < top_n; i++) {
          int idx = indices[i];
          std::cout << "  " << (i + 1) << ". Class " << idx << ": "
                    << std::fixed << std::setprecision(6)
                    << tmp_output[idx] * 100 << "%" << '\n';
        }

        int max_class = indices[0];
        float max_prob = tmp_output[max_class];
        std::cout << "Image: " << fs::path(image_path).filename().string()
                  << " -> Predicted digit: " << max_class
                  << " (probability: " << std::fixed << std::setprecision(6)
                  << max_prob * 100 << "%)" << '\n';

      } else {
        it_lab_ai::Tensor input = prepare_image(image, input_shape, model_name);

        size_t output_classes = 1000;
        it_lab_ai::Tensor output({1, output_classes}, it_lab_ai::Type::kFloat);

        Graph graph;
        build_graph(graph, input, output, json_path, options, false);

        std::cout << "Starting inference..." << '\n';
        try {
          graph.inference(options);
          std::cout << "Inference completed successfully." << '\n';
        } catch (const std::exception& e) {
          std::cerr << "ERROR during inference: " << e.what() << '\n';
        }
        print_time_stats(graph);
        std::vector<float> tmp_output =
            process_model_output(*output.as<float>(), model_name);

        int top_n = std::min(5, static_cast<int>(tmp_output.size()));
        std::vector<int> indices(tmp_output.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(
            indices.begin(), indices.begin() + top_n, indices.end(),
            [&](int a, int b) { return tmp_output[a] > tmp_output[b]; });

        std::cout << "Top " << top_n << " predictions:" << '\n';
        for (int i = 0; i < top_n; i++) {
          int idx = indices[i];
          std::cout << "  " << (i + 1) << ". Class " << idx << ": "
                    << std::fixed << std::setprecision(6) << tmp_output[idx];

          if (class_names.find(idx) != class_names.end()) {
            std::cout << " (" << class_names[idx] << ")";
          }
          std::cout << '\n';
        }

        int max_class = indices[0];
        float max_prob = tmp_output[max_class];
        std::cout << "Image: " << fs::path(image_path).filename().string()
                  << " -> Predicted class: " << max_class;
        if (class_names.find(max_class) != class_names.end()) {
          std::cout << " (" << class_names[max_class] << ")";
        }
        std::cout << " (probability: " << std::fixed << std::setprecision(6)
                  << max_prob << ")" << '\n';
      }
      std::cout << "----------------------------------------" << '\n';
    } catch (const std::exception& e) {
      std::cerr << "Error processing image " << image_path << ": " << e.what()
                << '\n';
    }
  }
  return 0;
}
