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

template <typename T>
bool elementwiseTensorCmp(const Tensor& t1, const Tensor& t2, double threshold) {
  if (t1.get_shape().count() != t2.get_shape().count()) return false;
  if (t1.get_type() != t2.get_type()) return false;
  
  size_t size = t1.get_shape().count();
  for (size_t i = 0; i < size; i++) {
    if (std::abs(*(t1.as<T>())[i] - *(t2.as<T>())[i]) > threshold) {
      return false;
    }
  }
  return true;
}
