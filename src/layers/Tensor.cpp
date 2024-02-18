#include <iostream>
#include <vector>
#include <ctime>
#include <iomanip>
#include <stdexcept>

#include "FCLayer.hpp"
#include "Tensor.hpp"

Tensor::Tensor(const Shape& sh, const std::vector<double>& values) {
  if (values.size() != sh.count())
    throw std::invalid_argument("vector length mismatch with shape");
  size_ = sh;
  values_ = values;
}

double& Tensor::operator()(const std::vector<size_t>& coords) {
  size_t s = size_.get_index(coords);
  return values_[s];
} // write

double Tensor::operator()(const std::vector<size_t>& coords) const {
  size_t s = size_.get_index(coords);
  return values_[s];
} // read

std::ostream& operator<<(std::ostream& out, const Tensor& t) {
    for (int i = 0; i < t.size_.count(); i++) {
      out.width(5);
      out << t.values_[i] << " ";
      if ((i + 1) % t.size_[1] == 0)
        out << std::endl;
    }

    return out;
}

Tensor initial_square_picture() {
    srand(time(nullptr));
    std::vector<size_t> initial_size = {224, 224};
    Tensor picture(initial_size);

    for (size_t h = 0; h < picture.get_size()[0]; h++) {
      for (size_t w = 0; w < picture.get_size()[1]; w++) {
        picture({h, w}) = rand() % 255;
      }
    }

    return picture;
}