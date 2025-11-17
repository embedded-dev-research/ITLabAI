#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "graph/graph.hpp"
#include "layers/ConvLayer.hpp"
#include "layers/EWLayer.hpp"
#include "layers/FCLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PoolingLayer.hpp"

using namespace it_lab_ai;

bool cmp_by_first(const std::pair<size_t, std::string>& a,
                  const std::pair<size_t, std::string>& b) {
  return a.first < b.first;
}

std::string generate_imgnet_val_string(size_t i) {
  std::string res = "ILSVRC2012_val_";
  std::string num = std::to_string(i);
  res = res + std::string(8 - num.size(), '0') + num + ".JPEG";
  return res;
}

std::vector<std::string> split(std::string str, const std::string& delim,
                               size_t count) {
  std::vector<std::string> result;
  size_t cur_count = 0;
  while (!str.empty() && cur_count < count) {
    size_t index = str.find(delim);
    if (index != std::string::npos) {
      result.push_back(str.substr(0, index));
      cur_count++;
      str = str.substr(index + delim.size());
    } else {
      result.push_back(str);
      cur_count++;
      str = "";
    }
  }
  return result;
}

size_t str_to_sizet(const std::string& inp) {
  size_t res = 0;
  if (inp.empty()) {
    return 0;
  }
  for (char i : inp) {
    res *= 10;
    res += i - '0';
  }
  return res;
}

Graph open_network(std::string path) {
  path += " ";
  return Graph(1);
}

void process_image(Tensor& input, const std::string& file) {
  size_t width = 227;
  cv::Mat image = cv::imread(file);
  if (image.empty()) {
    throw std::runtime_error("Failed to load image");
  }
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size((int)width, (int)width));
  std::vector<cv::Mat> channels;
  cv::split(resized_image, channels);
  std::vector<float> res(width * width * 3);
  int c = 0;
  for (int i = 0; i < (int)width; ++i) {
    for (int j = 0; j < (int)width; ++j) {
      res[c] = channels[2].at<uchar>(i, j);
      c++;
      res[c] = channels[1].at<uchar>(i, j);
      c++;
      res[c] = channels[0].at<uchar>(i, j);
      c++;
    }
  }
  if (input.get_shape().dims() > 0) {
    Shape sh({input.get_shape()[0] + 1, width, width, 3});
    std::vector<float> cur_input(*input.as<float>());
    cur_input.insert(cur_input.end(), res.begin(), res.end());
    input = make_tensor<float>(cur_input, sh);
  } else {
    Shape sh({1, width, width, 3});
    input = make_tensor<float>(res, sh);
  }
}

std::vector<std::pair<size_t, std::string> > extract_csv(
    const std::string& reference_path) {
  size_t n = 50000;
  std::ifstream ref;
  std::vector<std::pair<size_t, std::string> > lines(n);
  ref.open(reference_path);
  ref.ignore(1000, '\n');
  char buf[1001];
  for (size_t i = 0; i < n; i++) {
    ref.ignore(18);
    ref.getline(buf, 1000, ',');
    lines[i].first = str_to_sizet(buf);
    ref.getline(buf, 1000, '\n');
    lines[i].second = std::string(buf);
  }
  ref.close();
  return lines;
}

void check_accuracy(const std::string& neural_network_path,
                    const std::string& dataset_path, size_t imgs_size,
                    const std::string& reference_path) {
  Graph a1 = open_network(neural_network_path);
  Tensor input;
  Tensor output;
  auto inlayer = std::make_shared<InputLayer>();
  auto outlayer = std::make_shared<OutputLayer>();
  // ?? warning from linux
  outlayer->setID(1);
  inlayer->setID(0);
  //
  size_t k = 5;
  for (size_t i = 0; i < imgs_size; i++) {
    process_image(input,
                  dataset_path + "\\" + generate_imgnet_val_string(i + 1));
  }
  a1.setInput(inlayer, input);
  a1.setOutput(outlayer, output);
  a1.inference();
  size_t eqs;
  std::vector<size_t> eqs_info(imgs_size);
  std::vector<std::pair<size_t, std::string> > csv_info =
      extract_csv(std::move(reference_path));
  std::sort(csv_info.begin(), csv_info.end(), cmp_by_first);
  std::vector<std::string> cur_ref_topk;
  std::vector<std::string> cur_our_topk;
  for (size_t i = 0; i < imgs_size; i++) {
    eqs = 0;
    cur_ref_topk = split(csv_info[i].second, " ", k);
    // cur_our_topk = outlayer.top_k(output, k).first;
    cur_our_topk = cur_ref_topk;
    for (size_t j = 0; j < k; j++) {
      cur_ref_topk[j] = "";
      if (cur_ref_topk[j] == cur_our_topk[j]) {
        eqs++;
      }
    }
    eqs_info[i] = eqs;
  }
  // tmp output
  std::ofstream tmp;
  std::string buf;
  tmp.open("log.txt");
  for (size_t i = 0; i < imgs_size; i++) {
    buf = generate_imgnet_val_string(csv_info[i].first) + "\n" +
          std::to_string(eqs_info[i]) + "\n";
    tmp.write(buf.c_str(), buf.size());
  }
  tmp.close();
}

int main(int argc, char* argv[]) {
  if (argc >= 4) {
    check_accuracy(argv[1], argv[2], str_to_sizet(argv[3]), argv[4]);
  } else {
    std::cerr << "No input\n";
  }
  return 0;
}
