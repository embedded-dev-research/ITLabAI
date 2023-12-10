#pragma once
#include <stdexcept>
#include <vector>


typedef double ValueType;

using namespace std;

vector<ValueType> mat_vec_mul(const vector<vector<ValueType> >& mat,
                              const vector<ValueType>& vec);

class FCLayer {
public:
  FCLayer(): inputValues(), outputValues(), weights(), bias() {
    inputSize = 0;
    outputSize = 0;
  }
  FCLayer(const vector<ValueType>& input,
          const vector<vector<ValueType> >& input_weights,
          const vector<ValueType>& input_bias);
  FCLayer& operator=(const FCLayer& sec);
  void load_input(const vector<ValueType>& input);
  void set_weight(size_t i, size_t j, const ValueType& value) {
    if (i >= outputSize || j >= inputSize) {
      throw out_of_range("Bad weight index");
    }
    weights[i][j] = value;
  }
  vector<ValueType> get_output() const noexcept { return outputValues; }
  ValueType get_weight(size_t i, size_t j) const {
    if (i >= outputSize || j >= inputSize) {
      throw out_of_range("Bad weight index");
    }
    return weights[i][j];
  }
  void set_bias(size_t i, const ValueType& value) {
    if (i >= outputSize) {
      throw out_of_range("Bad bias index");
    }
    bias[i] = value;
  }
  ValueType get_bias(size_t i) const {
    if (i >= outputSize) {
      throw out_of_range("Bad bias index");
    }
    return bias[i];
  }
  void run();
 private:
  vector<ValueType> inputValues;
  size_t inputSize;
  vector<ValueType> outputValues;
  size_t outputSize;
  vector<vector<ValueType> > weights;
  vector<ValueType> bias;
};

// weights * inputValeus + bias = outputValues
