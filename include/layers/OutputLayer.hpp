#pragma once
#include <cmath>
#include <string>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class OutputLayer : public Layer {
 public:
  OutputLayer() = default;
  OutputLayer(const std::vector<std::string>& labels) : labels_(labels) {}
  static std::string get_name() { return "Output layer"; }
  void run(const Tensor& input, Tensor& output) override { output = input; }
  std::vector<std::string> get_labels() const { return labels_; }
  std::pair<std::vector<std::string>, Tensor> top_k(const Tensor& input,
                                                    size_t k) const;

 private:
  std::vector<std::string> labels_;
};

template <typename ValueType>
std::vector<ValueType> softmax(const std::vector<ValueType>& vec);

template <typename ValueType>
bool compare_pair(std::pair<std::string, ValueType> a,
                  std::pair<std::string, ValueType> b);

template <typename ValueType>
std::pair<std::vector<std::string>, std::vector<ValueType>> top_k_vec(
    const std::vector<ValueType>& input, const std::vector<std::string>& labels,
    size_t k);

}  // namespace itlab_2023
