#pragma once
#include <stdexcept>
#include <vector>

template<typename T>
class Shape2D {
 public:
  Shape2D() : data_(), width_(), height_() {}
  Shape2D(size_t height, size_t width = 1) : data_(width*height) {
    this->width_ = width;
    this->height_ = height;
  }
  Shape2D(size_t height, size_t width, const std::vector<T>& data)
      : data_(data) {
    this->width_ = width;
    this->height_ = height;
    data_.resize(width_ * height_, T(0));
  }
  Shape2D(const std::vector<T>& data) : data_(data), width_(1), height_(data.size()) {}
  Shape2D(const Shape2D &c) : data_(c.data_) {
    this->width_ = c.width_;
    this->height_ = c.height_;
  }
  Shape2D& operator=(const Shape2D &c) {
    this->data_ = c.data_;
    this->width_ = c.width_;
    this->height_ = c.height_;
    return *this;
  }
  T get(size_t i, size_t j = 0) const {
    if (i >= height_ || j >= width_) {
      throw std::out_of_range("Bad shape index");
    }
    return data_[i * width_ + j];
  }
  void set(size_t i, size_t j, const T& value) {
    if (i >= height_ || j >= width_) {
      throw std::out_of_range("Bad shape index");
    }
    data_[i * width_ + j] = value;
  }
  size_t get_width() const { return width_; }
  size_t get_height() const { return height_; }
  size_t size() const { return width_ * height_; }
  void resize(size_t height, size_t width) {
    this->width_ = width;
    this->height_ = height;
    data_.resize(width_ * height_, T(0));
  }
 private:
  std::vector<T> data_;
  size_t width_;
  size_t height_;
};

template <typename ValueType>
Shape2D<ValueType> mat_vec_mul(const Shape2D<ValueType>& mat,
                               const Shape2D<ValueType>& vec) {
  Shape2D<ValueType> res(mat.get_height());
  ValueType elem;
  for (size_t i = 0; i < mat.get_height(); i++) {
    elem = ValueType(0);
    for (size_t j = 0; j < vec.get_height(); j++) {
      elem += mat.get(i, j) * vec.get(j, 0);
    }
    res.set(i, 0, elem);
  }
  return res;
}

template <typename ValueType>
class Layer {
 public:
  virtual Shape2D<ValueType> run(
      const Shape2D<ValueType>& input) const = 0;
  size_t get_input_size() const { return inputSize_; }
  size_t get_output_size() const { return outputSize_; }
  // weights width x height
  std::pair<size_t, size_t> get_dims() const {
    return std::pair<size_t, size_t>(outputSize_, inputSize_);
  }

 protected:
  size_t inputSize_;
  size_t outputSize_;
};

template <typename ValueType>
class FCLayer : public Layer<ValueType> {
 public:
  FCLayer() : weights_(), bias_() {
    this->inputSize_ = 0;
    this->outputSize_ = 0;
  };
  FCLayer(const Shape2D<ValueType>& input_weights,
          const Shape2D<ValueType>& input_bias);
  FCLayer& operator=(const FCLayer& sec);
  void set_weight(size_t i, size_t j, const ValueType& value) {
    weights_.set(i, j, value);
  }
  ValueType get_weight(size_t i, size_t j) const {
    return weights_.get(i, j);
  }
  void set_bias(size_t i, const ValueType& value) {
    bias_.set(i, 0, value);
  }
  ValueType get_bias(size_t i) const { return bias_.get(i, 0); }
  Shape2D<ValueType> run(const Shape2D<ValueType>& input) const;

 private:
  Shape2D<ValueType> weights_;
  Shape2D<ValueType> bias_;
};

// weights * inputValues + bias = outputValues

// constructor for FCLayer
template <typename ValueType>
FCLayer<ValueType>::FCLayer(
    const Shape2D<ValueType>& input_weights,
    const Shape2D<ValueType>& input_bias)
    : weights_(input_weights), bias_(input_bias) {
  if (input_weights.size() == 0) {
    throw std::invalid_argument("Empty weights for FCLayer");
  }
  this->inputSize_ = input_weights.get_width();
  this->outputSize_ = input_bias.get_height();
  if (this->inputSize_ == 0 || this->outputSize_ == 0) {
    throw std::invalid_argument("Bad weights/bias size for FCLayer");
  }
  // make weights isize x osize, filling empty with 0s
  weights_.resize(this->outputSize_, this->inputSize_);
  //
}

template <typename ValueType>
FCLayer<ValueType>& FCLayer<ValueType>::operator=(const FCLayer& sec) {
  this->inputSize_ = sec.inputSize_;
  this->outputSize_ = sec.outputSize_;
  weights_ = sec.weights_;
  bias_ = sec.bias_;
  return *this;
}

template <typename ValueType>
Shape2D<ValueType> FCLayer<ValueType>::run(
    const Shape2D<ValueType>& input) const {
  if (this->outputSize_ == 0 || this->inputSize_ == 0) {
    throw std::runtime_error("FCLayer wasn't initialized normally");
  }
  if (input.get_height() != this->inputSize_) {
    throw std::invalid_argument("Input size doesn't fit FCLayer");
  }
  Shape2D<ValueType> output_values = mat_vec_mul(weights_, input);
  for (size_t i = 0; i < this->outputSize_; i++) {
    output_values.set(i, 0, output_values.get(i, 0) + bias_.get(i, 0));
  }
  return output_values;
}
