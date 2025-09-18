#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

class OutputLayer : public Layer {
 public:
  OutputLayer() = default;
  OutputLayer(const std::vector<std::string>& labels) : labels_(labels) {}
  static std::string get_name() { return "Output layer"; }
  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override {
    output = input;
  }
  std::vector<std::string> get_labels() const { return labels_; }
  std::pair<std::vector<std::string>, Tensor> top_k(const Tensor& input,
                                                    size_t k) const;
#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    Tensor a = make_tensor(v);
    return a;
  }
#endif

 private:
  std::vector<std::string> labels_;
};

template <typename ValueType>
std::vector<ValueType> softmax(const std::vector<ValueType>& vec) {
  if (vec.empty()) {
    throw std::invalid_argument("Empty vector in softmax");
  }
  ValueType max_elem = *std::max_element(vec.begin(), vec.end());
  std::vector<ValueType> res = vec;
  for (size_t i = 0; i < res.size(); i++) {
    res[i] = std::exp(res[i] - max_elem);  // <= 1
  }
  ValueType sum = std::accumulate(res.begin(), res.end(), ValueType(0));
  for (size_t i = 0; i < res.size(); i++) {
    res[i] /= sum;
  }
  return res;
}

template <typename ValueType>
std::vector<std::vector<ValueType>> softmax(
    const std::vector<ValueType>& fullvec, size_t c) {
  if (fullvec.empty()) {
    throw std::invalid_argument("Empty vector in softmax");
  }
  if (c == 0) {
    throw std::invalid_argument("c cannot be zero");
  }
  if (fullvec.size() % c != 0) {
    throw std::invalid_argument("Vector size must be divisible by c");
  }
  size_t p = fullvec.size() / c;
  std::vector<std::vector<ValueType>> fullres;
  for (size_t n = 0; n < p; n++) {
    std::vector<ValueType> vec(c);
    for (size_t row = 0; row < c; row++) {
      vec[row] = fullvec[n * c + row];
    }

    ValueType max_elem = *std::max_element(vec.begin(), vec.end());
    std::vector<ValueType> res = vec;
    for (size_t i = 0; i < res.size(); i++) {
      res[i] = std::exp(res[i] - max_elem);  // <= 1
    }
    ValueType sum = std::accumulate(res.begin(), res.end(), ValueType(0));
    for (size_t i = 0; i < res.size(); i++) {
      res[i] /= sum;
    }
    fullres.push_back(res);
  }
  return fullres;
}

template <typename ValueType>
bool compare_pair(std::pair<std::string, ValueType> a,
                  std::pair<std::string, ValueType> b) {
  return (a.second > b.second);
}

template <typename ValueType>
std::pair<std::vector<std::string>, std::vector<ValueType>> top_k_vec(
    const std::vector<ValueType>& input, const std::vector<std::string>& labels,
    size_t k) {
  if (input.size() != labels.size()) {
    throw std::invalid_argument("Labels size not equal input size");
  }
  if (k > input.size()) {
    throw std::invalid_argument("K cannot be bigger than input size");
  }
  std::vector<std::pair<std::string, ValueType>> sort_buf(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    sort_buf[i] = std::make_pair(labels[i], input[i]);
  }
  std::partial_sort(sort_buf.begin(), sort_buf.begin() + k, sort_buf.end(),
                    compare_pair<ValueType>);
  std::vector<std::string> res_labels(k);
  std::vector<ValueType> res_input(k);
  for (size_t i = 0; i < k; i++) {
    res_labels[i] = sort_buf[i].first;
    res_input[i] = sort_buf[i].second;
  }
  return std::make_pair(res_labels, res_input);
}

}  // namespace it_lab_ai
