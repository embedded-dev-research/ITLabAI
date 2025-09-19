#include "layers/FCLayer.hpp"

namespace it_lab_ai {

void FCLayer::run(const std::vector<Tensor>& input,
                  std::vector<Tensor>& output) {
  if (input.size() != 1) {
    throw std::runtime_error("FCLayer: Input tensors not 1");
  }
  if (input[0].get_type() != weights_.get_type()) {
    throw std::invalid_argument("input[0] and weights data type aren't same");
  }
  if (bias_.get_type() != weights_.get_type()) {
    throw std::invalid_argument("Bias and weights data type aren't same");
  }

  // Получаем batch_size и output_size
  size_t batch_size = input[0].get_shape()[0];
  size_t output_size = bias_.get_shape()[0];

  // Добавляем отладочные выводы
  std::cout << "FCLayer DEBUG:" << std::endl;
  std::cout << "  Input shape: ";
  for (size_t d = 0; d < input[0].get_shape().dims(); ++d) {
    std::cout << input[0].get_shape()[d] << " ";
  }
  std::cout << std::endl;

  std::cout << "  Weights shape: ";
  for (size_t d = 0; d < weights_.get_shape().dims(); ++d) {
    std::cout << weights_.get_shape()[d] << " ";
  }
  std::cout << std::endl;

  std::cout << "  Bias shape: ";
  for (size_t d = 0; d < bias_.get_shape().dims(); ++d) {
    std::cout << bias_.get_shape()[d] << " ";
  }
  std::cout << std::endl;

  std::cout << "  Batch size: " << batch_size << std::endl;
  std::cout << "  Output size: " << output_size << std::endl;

  switch (input[0].get_type()) {
    case Type::kInt: {
      FCLayerImpl<int> used_impl(*weights_.as<int>(), weights_.get_shape(),
                                 *bias_.as<int>());

      // Добавляем отладочный вывод перед вызовом run
      std::cout << "  Running INT implementation" << std::endl;

      auto result = used_impl.run(*input[0].as<int>());

      // Добавляем отладочный вывод после вычислений
      std::cout << "  Result vector size: " << result.size() << std::endl;
      std::cout << "  Expected output shape: [" << batch_size << ", "
                << output_size << "]" << std::endl;
      std::cout << "  Expected total elements: " << batch_size * output_size
                << std::endl;

      // Проверяем размер результата
      if (result.size() != batch_size * output_size) {
        throw std::runtime_error("Result size mismatch: got " +
                                 std::to_string(result.size()) + ", expected " +
                                 std::to_string(batch_size * output_size));
      }

      output[0] = make_tensor(result, {batch_size, output_size});
      break;
    }
    case Type::kFloat: {
      FCLayerImpl<float> used_impl(*weights_.as<float>(), weights_.get_shape(),
                                   *bias_.as<float>());

      // Добавляем отладочный вывод перед вызовом run
      std::cout << "  Running FLOAT implementation" << std::endl;

      auto result = used_impl.run(*input[0].as<float>());

      // Добавляем отладочный вывод после вычислений
      std::cout << "  Result vector size: " << result.size() << std::endl;
      std::cout << "  Expected output shape: [" << batch_size << ", "
                << output_size << "]" << std::endl;
      std::cout << "  Expected total elements: " << batch_size * output_size
                << std::endl;

      // Проверяем размер результата
      if (result.size() != batch_size * output_size) {
        throw std::runtime_error("Result size mismatch: got " +
                                 std::to_string(result.size()) + ", expected " +
                                 std::to_string(batch_size * output_size));
      }

      output[0] = make_tensor(result, {batch_size, output_size});
      break;
    }
    default: {
      throw std::runtime_error("No such type");
    }
  }

  std::cout << "  FCLayer completed successfully" << std::endl;
}

}  // namespace it_lab_ai
