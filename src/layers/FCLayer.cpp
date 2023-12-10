#include "layers/FCLayer.hpp"

vector<ValueType> mat_vec_mul(const vector<vector<ValueType> > &mat,
                              const vector<ValueType>& vec) {
  vector<ValueType> res(mat.size());
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

FCLayer::FCLayer(const vector<ValueType>& input,
        const vector<vector<ValueType> >& input_weights,
        const vector<ValueType>& input_bias)
    : inputValues(input),
      outputValues(input_bias.size()),
      weights(input_weights),
      bias(input_bias) {
  inputSize = input.size();
  outputSize = input_bias.size();
  // make weights isize x osize
  for (size_t i = 0; i < weights.size(); i++) {
    while (weights[i].size() < inputSize) {
      weights[i].push_back(ValueType(0));
    }
  }
  const vector<ValueType> empty(inputSize, ValueType(0));
  while (weights.size() < outputSize) {
    weights.push_back(empty);
  }
  //
}

FCLayer& FCLayer::operator=(const FCLayer& sec) {
  inputSize = sec.inputSize;
  inputValues = sec.inputValues;
  outputSize = sec.outputSize;
  outputValues = vector<ValueType>(outputSize);
  weights = sec.weights;
  bias = sec.bias;
  return *this;
}

void FCLayer::load_input(const vector<ValueType>& input) {
  if (inputSize == 0) {
    throw runtime_error("Layer wasn't initialized normally");
  }
  if (input.size() != inputSize) {
    throw invalid_argument("Incorrect input size");
  }
  inputValues = input;
}

void FCLayer::run() {
  if (outputSize == 0 || inputSize == 0) {
    throw runtime_error("Layer wasn't initialized normally");
  }
  outputValues = mat_vec_mul(weights, inputValues);
  for (size_t i = 0; i < outputSize; i++) {
    outputValues[i] += bias[i];
  }
}
