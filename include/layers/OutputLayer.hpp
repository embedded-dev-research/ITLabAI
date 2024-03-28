#pragma once
#include <cmath>
#include <string>

#include "layers/Layer.hpp"

class OutputLayer : public Layer {
 public:
  OutputLayer() = default;
  OutputLayer(const std::vector<std::string>& labels) : labels_(labels) {}
  static std::string get_name() { return "Output layer"; }
  static void run(const Tensor& input, Tensor& output) { output = input; }
  std::vector<std::string> get_labels() const { return labels_; }
  std::pair<std::vector<std::string>, Tensor> top_k(const Tensor& input,
                                                    size_t k) const;

 private:
  std::vector<std::string> labels_;
};

template <typename ValueType>
std::vector<ValueType> softmax(const std::vector<ValueType>& vec) {
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
bool compare_pair(std::pair<std::string, ValueType> a,
                  std::pair<std::string, ValueType> b) {
  return (a.second > b.second);
}

template <typename ValueType>
std::pair<std::vector<std::string>, std::vector<ValueType> > top_k_vec(
    const std::vector<ValueType>& input, const std::vector<std::string>& labels,
    size_t k) {
  if (input.size() != labels.size()) {
    throw std::invalid_argument("Labels size not equal input size");
  }
  if (k > input.size()) {
    throw std::invalid_argument("K cannot be bigger than input size");
  }
  // sort values in descending order
  std::vector<std::pair<std::string, ValueType> > sort_buf(input.size());
  for (size_t i = 0; i < input.size(); i++) {
    sort_buf[i] = make_pair(labels[i], input[i]);
  }
  std::sort(sort_buf.begin(), sort_buf.end(), compare_pair<ValueType>);
  // split vector of pairs to pairs of vectors
  std::vector<std::string> res_labels(k);
  std::vector<ValueType> res_input(k);
  for (size_t i = 0; i < k; i++) {
    res_labels[i] = sort_buf[i].first;
    res_input[i] = sort_buf[i].second;
  }
  return make_pair(res_labels, res_input);
}
