#include "layers/Tensor.hpp"

std::ostream& operator<<(std::ostream& out, const Tensor& t) {
  for (size_t i = 0; i < t.get_shape().count(); i++) {
    out.width(5);
    if (t.get_type() == Type::kInt) {
      out << (*t.as<int>())[i] << " ";
    } else if (t.get_type() == Type::kFloat) {
      out << (*t.as<float>())[i] << " ";
    }
    if (t.get_shape().dims() > 1) {
      if ((i + 1) % t.get_shape()[1] == 0) out << std::endl;
    }
  }
  return out;
}