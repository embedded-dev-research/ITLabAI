#pragma once

#include <iostream>
#include <vector>

#include "FCLayer.hpp"

class Tensor {
private:
  Shape size_;
  std::vector<double> values_;
public:
  Tensor() = default;
  Tensor(const size_t dims_count) : size_(dims_count) {values_ = {};};
  Tensor(const std::vector<size_t> &dims) : size_(dims) {values_ = {};};
  Tensor(const Shape& sh);
  Tensor(const Tensor& t) = default;

  Shape Get_size() const { return size_; }

  double& operator()(const std::vector<size_t>& coords); // write
  double operator()(const std::vector<size_t>& coords) const; // read

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};

