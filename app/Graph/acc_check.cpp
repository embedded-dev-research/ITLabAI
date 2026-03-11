
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#include <crtdbg.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_map>

#include "build.hpp"

class MemoryLogger {
 private:
  std::chrono::steady_clock::time_point start_time;
  size_t peak_memory = 0;
  size_t initial_memory = 0;

  size_t getProcessMemory() {
    HANDLE hProcess = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);

    if (GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc))) {
      return pmc.WorkingSetSize / (1024 * 1024);
    }
    return 0;
  }

 public:
  MemoryLogger() {
    start_time = std::chrono::steady_clock::now();
    initial_memory = getProcessMemory();
    log("START");
  }

  void log(const char* stage) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(now - start_time)
            .count();

    size_t current = getProcessMemory();
    if (current > peak_memory) peak_memory = current;

    std::cout << "[" << std::setw(4) << elapsed << "s] " << std::setw(30)
              << stage << " | "
              << "PROCESS MEM: " << std::setw(6) << current << " MB"
              << " (PEAK: " << std::setw(6) << peak_memory << " MB)"
              << " (DELTA: " << std::setw(4) << (current - initial_memory)
              << " MB)\n";
  }

  ~MemoryLogger() {
    log("END");
    std::cout << "====================================\n";
    std::cout << "PEAK PROCESS MEMORY: " << peak_memory << " MB\n";
    std::cout << "INITIAL PROCESS MEMORY: " << initial_memory << " MB\n";
    std::cout << "FINAL PROCESS MEMORY: " << getProcessMemory() << " MB\n";
    if (getProcessMemory() > initial_memory + 10) {
      std::cout << "WARNING: Process memory growth! (+"
                << (getProcessMemory() - initial_memory) << " MB)\n";
    } else {
      std::cout << "OK: No significant process memory growth\n";
    }
  }
};

MemoryLogger g_memLogger;

#define LOG_MEM(stage) g_memLogger.log(stage)

namespace fs = std::filesystem;
using namespace it_lab_ai;

int main(int argc, char* argv[]) {
  LOG_MEM("Program start");

  std::string model_name = "alexnet_mnist";
  RuntimeOptions options;
  size_t num_photo = 1000;
  size_t batch_size = 32;

  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--model" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (std::string(argv[i]) == "--batch" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
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
        num_photo = std::stoi(argv[i]);

        if (num_photo < 1 || num_photo > 50000) {
          std::cerr << "Warning: num_photo should be between 1 and 10000 "
                    << "Using value: " << num_photo << '\n';
        }
      } catch (const std::exception& e) {
        std::cerr << "Error: Invalid numeric argument: " << argv[i]
                  << ". Using default value: 1000" << e.what() << '\n';
      }
    }
  }

  LOG_MEM("After args parsing");

  std::string dataset_path;
  if (model_name == "alexnet_mnist") {
    dataset_path = MNIST_PATH;
  } else {
    dataset_path = IMAGENET_ACC;
  }

  std::string json_path = model_paths[model_name];
  std::vector<int> input_shape = get_input_shape_from_json(json_path);

  std::cout << '\n';
  int batch_count = 0;
  if (model_name == "alexnet_mnist") {
    LOG_MEM("MNIST start");

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
            for (size_t n = 0; n < name; n++) a += counts[n] + 1;
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
        for (size_t n = 0; n < name; n++) a += counts[n] + 1;
        if (name == indices[a]) stat++;
      }
    }
    double percentage =
        (static_cast<double>(stat) / static_cast<double>(sum + 10)) * 100;
    std::cout << "Stat: " << std::fixed << std::setprecision(2) << percentage
              << "%" << '\n';

    LOG_MEM("MNIST end");
    return 0;
  }

  LOG_MEM("ImageNet start");

  std::vector<size_t> counts(1000, 0);
  std::vector<std::string> image_paths;
  std::vector<int> true_labels;
  std::vector<float> all_image_data;
  size_t total_images = 0;

  LOG_MEM("Counting classes");
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
        }
      }
    }
  }

  size_t images_per_class_base = num_photo / 1000;
  size_t remaining = num_photo % 1000;

  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  size_t image_size = channels * height * width;
  size_t output_classes = 1000;

  LOG_MEM("Reserving memory");
  all_image_data.reserve(num_photo * image_size);
  image_paths.reserve(num_photo);
  true_labels.reserve(num_photo);

  total_images = 0;

  LOG_MEM("Loading images start");
  for (int class_id = 0; class_id < 1000; ++class_id) {
    size_t need_from_class = images_per_class_base;
    if (remaining > 0) {
      need_from_class++;
      remaining--;
    }

    if (need_from_class == 0) continue;

    std::ostringstream folder_oss;
    folder_oss << std::setw(5) << std::setfill('0') << class_id;
    std::string class_folder_path = dataset_path + "/" + folder_oss.str();

    if (!fs::exists(class_folder_path)) continue;

    size_t taken = 0;
    for (const auto& entry : fs::directory_iterator(class_folder_path)) {
      if (taken >= need_from_class) break;

      if (entry.path().extension() == ".png" ||
          entry.path().extension() == ".jpg" ||
          entry.path().extension() == ".jpeg") {
        cv::Mat image = cv::imread(entry.path().string());
        if (image.empty()) {
          std::cerr << "Failed to load image: " << entry.path().string()
                    << '\n';
          continue;
        }

        it_lab_ai::Tensor prepared_tensor =
            prepare_image(image, input_shape, model_name);
        const std::vector<float>& image_data = *prepared_tensor.as<float>();

        all_image_data.insert(all_image_data.end(), image_data.begin(),
                              image_data.end());

        image_paths.push_back(entry.path().string());
        true_labels.push_back(class_id);
        taken++;
        total_images++;
      }
    }

    if (taken < need_from_class) {
      std::cout << "Warning: Class " << class_id << " has only " << taken
                << " images (needed " << need_from_class << ")" << '\n';
    }

    if (class_id % 100 == 0 && class_id > 0) {
      char buf[50];
      sprintf(buf, "Class %d", class_id);
      LOG_MEM(buf);
    }
  }

  LOG_MEM("Images loaded");

  if (total_images != num_photo) {
    std::cout << "Warning: Requested " << num_photo << " images but loaded "
              << total_images << " due to insufficient data" << '\n';
    num_photo = total_images;
  }

  int correct_predictions_top1 = 0;
  int correct_predictions_top5 = 0;

  LOG_MEM("Starting batch processing");
  auto total_start_time = std::chrono::high_resolution_clock::now();
  int total_inference_time = 0;

  for (size_t batch_start = 0; batch_start < num_photo;
       batch_start += batch_size) {
    size_t batch_end = std::min(batch_start + batch_size, num_photo);
    size_t current_batch_size = batch_end - batch_start;

    char batch_log[100];
    sprintf(batch_log, "Batch %zu/%zu (size %zu)", batch_start / batch_size + 1,
            (num_photo + batch_size - 1) / batch_size, current_batch_size);
    LOG_MEM(batch_log);

    std::vector<float> batch_data;
    batch_data.reserve(current_batch_size * image_size);

    size_t batch_offset = batch_start * image_size;
    batch_data.insert(batch_data.end(), all_image_data.begin() + batch_offset,
                      all_image_data.begin() + batch_offset +
                          current_batch_size * image_size);

    it_lab_ai::Shape batch_input_shape(
        {current_batch_size, static_cast<size_t>(channels),
         static_cast<size_t>(height), static_cast<size_t>(width)});
    it_lab_ai::Tensor batch_input = make_tensor(batch_data, batch_input_shape);

    it_lab_ai::Shape batch_output_shape({current_batch_size, output_classes});
    it_lab_ai::Tensor batch_output(batch_output_shape, it_lab_ai::Type::kFloat);

    Graph graph;
    build_graph(graph, batch_input, batch_output, json_path, options, false);

    LOG_MEM("Batch inference");
    // auto batch_start_time =
    //     std::chrono::high_resolution_clock::now();
    graph.inference(options);
    total_inference_time += print_time_stats(graph);
    // auto batch_end_time = std::chrono::high_resolution_clock::now();
    // int batch_time =
    //     static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
    //                          batch_end_time - batch_start_time)
    //                          .count());  // ← Добавлен static_cast
    // total_inference_time += batch_time;
    // batch_count++;

    // #ifdef ENABLE_STATISTIC_TIME
    //          std::vector<int> elps_time = graph.getTime();
    //          int batch_time = std::accumulate(elps_time.begin(),
    //          elps_time.end(), 0); total_inference_time += batch_time;
    //          batch_count++;
    //
    //          char time_log[100];
    //          sprintf(time_log, "Batch %d time: %d ms", batch_count,
    //          batch_time); LOG_MEM(time_log);
    // #endif

    const std::vector<float>& raw_batch_output = *batch_output.as<float>();

    for (size_t i = 0; i < current_batch_size; ++i) {
      size_t global_idx = batch_start + i;

      std::vector<float> single_output(
          raw_batch_output.begin() + i * output_classes,
          raw_batch_output.begin() + (i + 1) * output_classes);

      float max_val =
          *std::max_element(single_output.begin(), single_output.end());
      float sum = 0.0f;
      for (float& val : single_output) {
        val = exp(val - max_val);
        sum += val;
      }
      for (float& val : single_output) {
        val /= sum;
      }

      std::vector<size_t> indices(single_output.size());
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return single_output[a] > single_output[b];
      });

      if (indices[0] == static_cast<size_t>(true_labels[global_idx])) {
        correct_predictions_top1++;
      }

      for (int top_k = 0; top_k < std::min(5, static_cast<int>(indices.size()));
           ++top_k) {
        if (indices[top_k] == static_cast<size_t>(true_labels[global_idx])) {
          correct_predictions_top5++;
          break;
        }
      }
    }

    batch_data.clear();
    batch_data.shrink_to_fit();
  }

  auto total_end_time = std::chrono::high_resolution_clock::now();
  int total_time =
      static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                           total_end_time - total_start_time)
                           .count());

  std::cout << "\n!INFERENCE TIME INFO START!" << '\n';
  std::cout << "Total inference time (sum of batches): " << total_inference_time
            << " ms\n";
  std::cout << "Total wall-clock time for all batches: " << total_time
            << " ms\n";
  std::cout << "Number of batches: " << batch_count << '\n';
  std::cout << "Average time per batch: "
            << (batch_count > 0 ? total_inference_time / batch_count : 0)
            << " ms\n";
  std::cout << "!INFERENCE TIME INFO END!" << '\n';
  /*std::cout << "\n!INFERENCE TIME INFO START!" << '\n';
  std::cout << "Total inference time for all batches: " << total_inference_time
            << " ms\n";
  std::cout << "Number of batches: " << batch_count << '\n';
  std::cout << "!INFERENCE TIME INFO END!" << '\n';
  LOG_MEM("All batches processed");*/

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

  all_image_data.clear();
  all_image_data.shrink_to_fit();
  image_paths.clear();
  image_paths.shrink_to_fit();
  true_labels.clear();
  true_labels.shrink_to_fit();

  LOG_MEM("Program end");
  return 0;
}