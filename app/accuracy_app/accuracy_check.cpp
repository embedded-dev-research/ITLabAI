#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

int main() {
  std::string image_directory = "./photos";

  std::vector<cv::String> image_paths;
  cv::glob(image_directory + "*.jpg", image_paths);
  cv::glob(image_directory + "*.png", image_paths);

  for (const auto& path : image_paths) {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
      std::cerr << "The image could not be opened: " << image_path << std::endl;
      continue;
    }

    // image processing will be here...
    // For check. This is temporary:
    std::cout << "Image size ::: " << image_path << ": " << image.size() << std::endl;
    }

  return 0;
}