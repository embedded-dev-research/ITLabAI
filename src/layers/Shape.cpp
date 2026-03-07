#include "layers/Shape.hpp"

namespace it_lab_ai {

size_t Shape::get_index(const std::vector<size_t>& coords) const {
  if (coords.size() != dims_.size()) {
    throw std::invalid_argument("Invalid index vector");
  }
  size_t res = 0;
  for (size_t i = 0; i < coords.size(); i++) {
    // to get to the i line
    const size_t mulbuf =
        std::accumulate(dims_.cbegin() + (i + 1), dims_.cend(),
                        static_cast<size_t>(1), std::multiplies<>());
    if (coords[i] >= dims_[i]) {
      throw std::out_of_range("Out of range");
    }
    res += coords[i] * mulbuf;
  }
  return res;
}
std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  for (size_t i = 0; i < shape.dims(); ++i) {
    os << shape[i];
    if (i < shape.dims() - 1) {
      os << " ";
    }
  }
  return os;
}

}  // namespace it_lab_ai
