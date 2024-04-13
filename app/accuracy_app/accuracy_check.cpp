#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

int main() {
  std::string directory = "./photos";

  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      cv::Mat image = cv::imread(entry.path().string());

      // For example:
      if (!image.empty()) {
        cv::imshow("Image", image);
        cv::waitKey(0);
      } else {
        std::cerr << "Read image error. Image path is: " << entry.path() << std::endl;
      }
    }
  }

  return 0;
}