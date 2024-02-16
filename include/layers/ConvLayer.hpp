#pragma once

#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>

#include "FCLayer.hpp"

class Tensor {
private:
  Shape size;
  std::vector<double> values;
public:
  Tensor() = default;
  Tensor(const size_t dims_count) : size(dims_count) {values = {};};
  Tensor(const std::vector<size_t> &dims) : size(dims) {values = {};};
  Tensor(const Shape& sh);
  Tensor(const Tensor& t) = default;

  Shape Get_size() const { return size; }

  double& operator()(const std::vector<size_t>& coords); // write
  double operator()(const std::vector<size_t>& coords) const; // read

  friend std::ostream& operator<<(std::ostream& out, const Tensor& t);
};

Tensor::Tensor(const Shape& sh) {
  size = sh;
  values = std::vector<double>(sh.count(), 0);
}

double& Tensor::operator()(const std::vector<size_t>& coords) {
  size_t s = size.get_index(coords);
  return values[s];
} // write

double Tensor::operator()(const std::vector<size_t>& coords) const {
  size_t s = size.get_index(coords);
  return values[s];
} // read

std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    for (int i = 0; i < t.size.count(); i++) {
      std::cout.width(5);
      out << t.values[i] << " ";
      if ((i + 1) % t.size[1] == 0)
        out << std::endl;
    }

    return out;
}

Tensor initial_square_picture() {
    srand(time(0));
    std::vector<size_t> initial_size = {224, 224};
    Tensor picture(initial_size);

    for (size_t h = 0; h < picture.Get_size()[0]; h++) {
      for (size_t w = 0; w < picture.Get_size()[1]; w++) {
        picture({h, w}) = rand() % 255;
      }
    }

    return picture;
}