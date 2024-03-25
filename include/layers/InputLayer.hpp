#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include "layers/Layer.hpp"

class InputLayer : public Layer {
 public:
  InputLayer() = default;
  void run(const std::vector<std::string>& path, Tensor& output) const {
    std::vector<int> res(path.size() * 227 * 227 * 3);
    for (int num = 0; num < path.size(); num++) {
      cv::Mat image = cv::imread(path[num]);
      if (image.empty()) {
        throw std::runtime_error("Failed to load image");
      }
      cv::Mat resized_image;
      cv::resize(image, resized_image, cv::Size(227, 227));
      std::vector<cv::Mat> channels;
      cv::split(resized_image, channels);
      int c = 0;
      for (int i = 0; i < 227; ++i) {
        for (int j = 0; j < 227; ++j) {
          res[c] = static_cast<int>(channels[2].at<uchar>(i, j));
          c++;
          res[c] = static_cast<int>(channels[1].at<uchar>(i, j));
          c++;
          res[c] = static_cast<int>(channels[0].at<uchar>(i, j));
          c++;
        }
      }
    }
    Shape sh({static_cast<size_t>(path.size()), 227, 227, 3});
    output = make_tensor<int>(res, sh);
  }
};
