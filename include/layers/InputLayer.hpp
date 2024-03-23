#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>

#include "layers/Layer.hpp"

template <typename ValueType>
class InputLayer : public Layer<ValueType> {
 private:
  int Count_pic_;

 public:
  InputLayer(const int& Cpic) : Layer<ValueType>() { Count_pic_ = Cpic; }
  std::vector<ValueType> run(const std::vector<ValueType>& input) const {
    std::vector<ValueType> a = input;
    return a;
  }
  Tensor run(const std::vector<std::string>& path) const {
    std::vector<int> res(Count_pic_ * 227 * 227 * 3);
    for (int num = 0; num < Count_pic_; num++) {
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
    Shape sh({static_cast<size_t>(Count_pic_), 227, 227, 3});
    const Tensor t = make_tensor<int>(res, sh);
    return t;
  }
};
