#pragma once
#include <stdexcept>
#include <vector>

template <typename ValueType>
std::vector<ValueType> mat_vec_mul(
    const std::vector<std::vector<ValueType> >& mat,
    const std::vector<ValueType>& vec) {
  std::vector<ValueType> res(mat.size());
  ValueType elem;
  for (size_t i = 0; i < mat.size(); i++) {
    elem = ValueType(0);
    for (size_t j = 0; j < vec.size(); j++) {
      elem += mat[i][j] * vec[j];
    }
    res[i] = elem;
  }
  return res;
}

template <typename ValueType>
class Layer {
 public:
  virtual std::vector<ValueType> run(
      const std::vector<ValueType>& input) const = 0;
};

template <typename ValueType>
class FCLayer : public Layer<ValueType> {
 public:
  FCLayer() : weights(), bias(), inputSize(0), outputSize(0) {}
  FCLayer(const std::vector<std::vector<ValueType> >& input_weights,
          const std::vector<ValueType>& input_bias);
  FCLayer& operator=(const FCLayer& sec);
  void set_weight(size_t i, size_t j, const ValueType& value) {
    if (i >= outputSize || j >= inputSize) {
      throw std::out_of_range("Bad weight index for FCLayer");
    }
    weights[i][j] = value;
  }
  ValueType get_weight(size_t i, size_t j) const {
    if (i >= outputSize || j >= inputSize) {
      throw std::out_of_range("Bad weight index for FCLayer");
    }
    return weights[i][j];
  }
  void set_bias(size_t i, const ValueType& value) {
    if (i >= outputSize) {
      throw std::out_of_range("Bad bias index for FCLayer");
    }
    bias[i] = value;
  }
  ValueType get_bias(size_t i) const {
    if (i >= outputSize) {
      throw std::out_of_range("Bad bias index for FCLayer");
    }
    return bias[i];
  }
  size_t get_input_size() const { return inputSize; }
  size_t get_output_size() const { return outputSize; }
  // weights width x height
  std::pair<size_t, size_t> get_dims() const {
    return std::pair<size_t, size_t>(outputSize, inputSize);
  }
  std::vector<ValueType> run(const std::vector<ValueType>& input) const;

 private:
  size_t inputSize;
  size_t outputSize;
  std::vector<std::vector<ValueType> > weights;
  std::vector<ValueType> bias;
};

// weights * inputValues + bias = outputValues

// constructor for FCLayer
template <typename ValueType>
FCLayer<ValueType>::FCLayer(
    const std::vector<std::vector<ValueType> >& input_weights,
    const std::vector<ValueType>& input_bias)
    : weights(input_weights), bias(input_bias) {
  if (input_weights.size() == 0) {
    throw std::invalid_argument("Empty weights for FCLayer");
  }
  inputSize = input_weights[0].size();
  outputSize = input_bias.size();
  if (inputSize == 0 || outputSize == 0) {
    throw std::invalid_argument("Bad weights/bias size for FCLayer");
  }
  // make weights isize x osize, filling empty with 0s
  for (size_t i = 0; i < weights.size(); i++) {
    weights[i].resize(inputSize, ValueType(0));
  }
  const std::vector<ValueType> empty(inputSize, ValueType(0));
  weights.resize(outputSize, empty);
  //
}

template <typename ValueType>
FCLayer<ValueType>& FCLayer<ValueType>::operator=(const FCLayer& sec) {
  inputSize = sec.inputSize;
  outputSize = sec.outputSize;
  weights = sec.weights;
  bias = sec.bias;
  return *this;
}

template <typename ValueType>
std::vector<ValueType> FCLayer<ValueType>::run(
    const std::vector<ValueType>& input) const {
  if (outputSize == 0 || inputSize == 0) {
    throw std::runtime_error("FCLayer wasn't initialized normally");
  }
  if (input.size() != inputSize) {
    throw std::invalid_argument("Input size doesn't fit FCLayer");
  }
  std::vector<ValueType> outputValues = mat_vec_mul(weights, input);
  for (size_t i = 0; i < outputSize; i++) {
    outputValues[i] += bias[i];
  }
  return outputValues;
}
