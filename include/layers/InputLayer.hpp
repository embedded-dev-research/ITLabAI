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
    std::vector<ValueType> a=input;
    return a;
  }
  std::vector<int> run(const std::string& path) const {
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
      throw std::runtime_error("Failed to load image");
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(227, 227));
    std::vector<cv::Mat> channels;
    cv::split(resized_image, channels);
    std::vector<std::vector<int>> red_values(227, std::vector<int>(227));
    std::vector<std::vector<int>> green_values(227, std::vector<int>(227));
    std::vector<std::vector<int>> blue_values(227, std::vector<int>(227));
    for (int i = 0; i < 227; ++i) {
      for (int j = 0; j < 227; ++j) {
        red_values[i][j] = static_cast<int>(channels[2].at<uchar>(i, j));
        green_values[i][j] = static_cast<int>(channels[1].at<uchar>(i, j));
        blue_values[i][j] = static_cast<int>(channels[0].at<uchar>(i, j));
      }
    }
    std::vector<int> res(227 * 227 * 3);
    int c = 0;
    for (int i = 0; i < 227; ++i) {
      for (int j = 0; j < 227; ++j) {
        res[c] = red_values[i][j];
        c++;
        res[c] = green_values[i][j];
        c++;
        res[c] = blue_values[i][j];
        c++;
      }
    }
    return res;
  }
};
