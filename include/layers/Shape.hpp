#pragma once

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace itlab_2023 {

class Shape {
 public:
  Shape() = default;
  Shape(size_t dims_count) : dims_(dims_count, 1) {}
  Shape(const std::vector<size_t>& dims) : dims_(dims) {}
  Shape(const std::initializer_list<size_t>& l) : dims_(l) {}
  Shape(const Shape& c) = default;
  Shape& operator=(const Shape& c) = default;
  size_t operator[](size_t i) const noexcept { return dims_[i]; }
  size_t& operator[](size_t i) noexcept { return dims_[i]; }
  size_t at(size_t i) const {
    if (i >= dims_.size()) {
      throw std::out_of_range("Invalid shape index");
    }
    return dims_[i];
  }
  size_t& at(size_t i) {
    if (i >= dims_.size()) {
      throw std::out_of_range("Invalid shape index");
    }
    return dims_[i];
  }
  void resize(const std::vector<size_t>& new_size) { dims_ = new_size; }
  size_t count() const {
    return std::accumulate(dims_.begin(), dims_.end(), size_t(1),
                           std::multiplies<>());
  }
  size_t dims() const noexcept { return dims_.size(); }
  size_t get_index(const std::vector<size_t>& coords) const;
  bool operator==(const Shape& other) const {
    if (dims_.size() != other.dims_.size()) return false;
    for (size_t i = 0; i < dims_.size(); ++i) {
      if (dims_[i] != other.dims_[i]) return false;
    }
    return true;
  }

  bool operator!=(const Shape& other) const { return !(*this == other); }
  friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

 private:
  std::vector<size_t> dims_;
};

}  // namespace itlab_2023
