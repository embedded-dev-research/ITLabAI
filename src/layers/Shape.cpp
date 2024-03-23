#include "layers/Shape.hpp"

size_t Shape::get_index(const std::vector<size_t>& coords) const {
  if (coords.size() != dims_.size()) {
    throw std::invalid_argument("Invalid index vector");
  }
  size_t res = 0;
  size_t mulbuf;
  for (size_t i = 0; i < coords.size(); i++) {
    // to get to the i line
    mulbuf = std::accumulate(dims_.begin() + (i + 1), dims_.end(), size_t(1),
                             std::multiplies<>());
    if (coords[i] >= dims_[i]) {
      throw std::out_of_range("Out of range");
    }
    res += coords[i] * mulbuf;
  }
  return res;
}
