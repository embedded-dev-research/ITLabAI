#pragma once

#include "Matrix.hpp"
#include <iostream>

struct hyperparameters {
	int height;
	int width;
};

class Conv {
	tensor_size result_size;
  tensor_size initial_size;
	hyperparameters padding;
	hyperparameters stride;

	Matrix initial_matrix;
  Matrix init_with_paddings;
	Matrix result_matrix;
  Matrix mask;

	Matrix add_padding();
  double mask_n_matrix_mul(int h, int w);
public:
	Conv(Matrix& input_tensor, Matrix& mask, hyperparameters padding = { 0, 0 }, hyperparameters stride = { 1, 1 });
  void run();
};

Conv::Conv(Matrix& input_tensor, Matrix& mask, hyperparameters padd, hyperparameters strd) {
	padding = padd;
	stride = strd;
  initial_size = input_tensor.Get_size();
  initial_matrix = input_tensor;

	// size parameters of input matrix
	int W_i = initial_size.width;
	int H_i = initial_size.height;

	// size parameters of mask matrix
	int w_m = mask.Get_size().width;
	int h_m = mask.Get_size().height;

	result_size.height = (H_i - h_m + 2 * padding.height) / stride.height + 1; // нужно ли учитывать как ошибку -- дробное число, как неправильно заданный stride???
	result_size.width = (W_i - w_m + 2 * padding.width) / stride.width + 1;

	result_matrix = Matrix(result_size.height, result_size.width);

	// change initial -- add paddings
  init_with_paddings = add_padding();
	std::cout << init_with_paddings;
}

double Conv::mask_n_matrix_mul(int h, int w) {
  double sum = 0.0;
  for (int i = h; i < mask.Get_size().height; i += stride.height) {
    for (int j = w; j < mask.Get_size().height; j += stride.width) {
      sum += mask(i - h, j - w) * init_with_paddings(i, j);
    }
  }
  return sum;
}

void Conv::run() {
  for (int h = 0; h < result_size.height; h++) {
    for (int w = 0; w < result_size.width; w++) {
      result_matrix(h, w) = mask_n_matrix_mul(h, w);
    }
  }
}

Matrix Conv::add_padding() {
	int width_with_padd = padding.width * 2 + initial_size.width;
	int height_with_padd = padding.height * 2 + initial_size.height;

	Matrix init_matrix_with_padd(height_with_padd, width_with_padd);

	for (int i = 0; i < initial_size.height; i++) {
		for (int j = 0; j < initial_size.width; j++) {
			init_matrix_with_padd(i + padding.height, j + padding.width) = initial_matrix(i, j);
		}
	}

	return init_matrix_with_padd;
}