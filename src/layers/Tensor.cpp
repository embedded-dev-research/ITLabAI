#include "layers/Tensor.hpp"

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

Tensor initial_square_int_picture() {
  srand((unsigned int)time(nullptr));
  std::vector<size_t> initial_size = {224, 224};
  Tensor picture(initial_size, kInt);

  for (size_t h = 0; h < picture.get_shape()[0]; h++) {
    for (size_t w = 0; w < picture.get_shape()[1]; w++) {
      picture.get<int>({h, w}) = rand() % 255;
    }
  }

  return picture;
}