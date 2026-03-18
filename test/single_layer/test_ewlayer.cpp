#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "fixture.hpp"
#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"

using namespace it_lab_ai;

class EWLayerTest_F : public BaseTestFixture {};

class EWTestsParameterized
    : public ::testing::TestWithParam<
          std::tuple<std::vector<double>, EWLayerImpl<double>,
                     std::vector<double>, std::function<double(double)>>> {};
// 1) input; 2) constructed ewlayerimpl; 3) expected_output; 4) lambda_expr.

TEST_P(EWTestsParameterized, element_wise_works_correctly) {
  auto data = GetParam();
  std::vector<double> input = std::get<0>(data);
  EWLayerImpl<double> a = std::get<1>(data);
  std::vector<double> output = a.run(input);
  std::vector<double> true_output = std::get<2>(data);
  auto func = std::get<3>(data);
  if (func != nullptr) {
    true_output = std::vector<double>(input.size());
    std::transform(input.begin(), input.end(), true_output.begin(), func);
  }
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(output[i], true_output[i], 1e-5);
  }
}

std::vector<double> basic_data1 = {2.0, 3.9, 0.1, 2.3};
std::vector<double> basic_data2 = {1.0, -1.0, 2.0, -2.0};

INSTANTIATE_TEST_SUITE_P(
    element_wise_tests, EWTestsParameterized,
    ::testing::Values(
        std::make_tuple(basic_data1, EWLayerImpl<double>({2, 2}, "minus"),
                        std::vector<double>({-2.0, -3.9, -0.1, -2.3}),
                        std::function<double(double)>()),
        std::make_tuple(basic_data1, EWLayerImpl<double>({2, 2}, "sin"),
                        std::vector<double>(),
                        std::function<double(double)>([](double arg) -> double {
                          return std::sin(arg);
                        })),
        std::make_tuple(basic_data2, EWLayerImpl<double>({2, 2}, "relu"),
                        std::vector<double>({1.0, 0.0, 2.0, 0.0}),
                        std::function<double(double)>()),
        std::make_tuple(basic_data2, EWLayerImpl<double>({2, 2}, "tanh"),
                        std::vector<double>(),
                        std::function<double(double)>([](double arg) -> double {
                          return std::tanh(arg);
                        })),
        std::make_tuple(basic_data2,
                        EWLayerImpl<double>({2, 2}, "linear", 2.0F, 1.0F),
                        std::vector<double>({3.0, -1.0, 5.0, -3.0}),
                        std::function<double(double)>()),
        std::make_tuple(std::vector<double>({0.0, 1.0, -1.0}),
                        EWLayerImpl<double>({3}, "sigmoid"),
                        std::vector<double>(),
                        std::function<double(double)>([](double x) {
                          return 1.0 / (1.0 + std::exp(-x));
                        })),
        std::make_tuple(std::vector<double>{-100.0, -50.0, 0.0, 50.0, 100.0},
                        EWLayerImpl<double>({5}, "sigmoid"),
                        std::vector<double>(),
                        std::function<double(double)>([](double x) {
                          if (x >= 0) {
                            double z = std::exp(-x);
                            return 1.0 / (1.0 + z);
                          } else {
                            double z = std::exp(x);
                            return z / (1.0 + z);
                          }
                        }))));

TEST(ewlayer, new_ewlayer_can_relu_float) {
  EWLayer layer("relu");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_mul_float) {
  EWLayer layer("linear", 2.0f, 0.0f);
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -5.0F});
  Tensor output;
  std::vector<float> converted_input = {2.0F, -2.0F, 4.0F, -10.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_sub_float) {
  EWLayer layer("linear", 1.0f, -1.0f);
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -5.0F});
  Tensor output;
  std::vector<float> converted_input = {0.0F, -2.0F, 1.0F, -6.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], converted_input[i], 1e-5);
  }
}

TEST(ewlayer, new_ewlayer_can_relu_int) {
  EWLayer layer("relu");
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output;
  std::vector<int> converted_input = {1, 0, 2, 0};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*out[0].as<int>())[i], converted_input[i]);
  }
}

TEST(ewlayer, new_ewlayer_can_linear_float) {
  EWLayer layer("linear", 2.0F, 3.0F);
  Tensor input = make_tensor<int>({1, -1, 2, -2});
  Tensor output = make_tensor<int>({0});
  std::vector<int> converted_input = {5, 1, 7, -1};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ((*out[0].as<int>())[i], converted_input[i]);
  }
}

TEST(ewlayer, IncompatibleInput) {
  EWLayer layer("abra");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  std::vector<Tensor> in{input, input};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(ewlayer, new_ewlayer_throws_with_invalid_function) {
  EWLayer layer("abra");
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F});
  Tensor output;
  std::vector<float> converted_input = {1.0F, 0.0F, 2.0F, 0.0F};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(ewlayer, new_ewlayer_can_sigmoid_float) {
  EWLayer layer("sigmoid");
  Tensor input = make_tensor<float>({0.0F, -1.0F, 1.0F, 2.0F});
  Tensor output;
  std::vector<float> expected_output = {0.5F, 1.0F / (1.0F + std::exp(1.0F)),
                                        1.0F / (1.0F + std::exp(-1.0F)),
                                        1.0F / (1.0F + std::exp(-2.0F))};
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  for (size_t i = 0; i < 4; i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], expected_output[i], 1e-5F);
  }
}

TEST(ewlayer, new_ewlayer_can_sigmoid_int) {
  EWLayer layer("sigmoid");
  Tensor input = make_tensor<int>({0, -100, 100, 1, -1});
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  std::vector<int> expected = {1, 0, 1, 1, 0};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ((*out[0].as<int>())[i], expected[i]);
  }
}

TEST(ewlayer, new_ewlayer_can_sigmoid_float_extreme_values) {
  EWLayer layer("sigmoid");
  Tensor input = make_tensor<float>({0.0F, -1.0F, 1.0F, 2.0F, -100.0F, 100.0F});
  Tensor output;

  auto stable_sigmoid = [](float x) {
    if (x >= 0) {
      float z = std::exp(-x);
      return 1.0F / (1.0F + z);
    } else {
      float z = std::exp(x);
      return z / (1.0F + z);
    }
  };

  std::vector<float> expected_output = {
      stable_sigmoid(0.0F), stable_sigmoid(-1.0F),   stable_sigmoid(1.0F),
      stable_sigmoid(2.0F), stable_sigmoid(-100.0F), stable_sigmoid(100.0F)};

  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);

  for (size_t i = 0; i < expected_output.size(); i++) {
    EXPECT_NEAR((*out[0].as<float>())[i], expected_output[i], 1e-5F);
  }
}

TEST_F(EWLayerTest_F, parallel_for_ew) {
  EWLayer layer("relu");

  std::vector<int> vec(8000000, -1);
  Tensor input = make_tensor<int>(vec);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp};

  for (auto backend : backends) {
    auto options = createOptionsWithBackend(backend);

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << " time: " << duration.count() << " ms" << std::endl;
    for (size_t i = 0; i < 8000000; i++) {
      EXPECT_EQ((*out[0].as<int>())[i], 0);
    }
  }
}

TEST(ewlayer, parallel_for_ew_sigmoid_compact) {
  EWLayer layer("sigmoid");

  std::vector<int> vec(8000000, -1);
  Tensor input = make_tensor<int>(vec);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<std::pair<ParBackend, std::string>> backends = {
      {ParBackend::kSeq, "Sequential"},
      {ParBackend::kThreads, "Threads"},
      {ParBackend::kTbb, "TBB"},
      {ParBackend::kOmp, "OpenMP"}};

  std::vector<int> reference_result;
  bool first = true;

  for (const auto& [backend, name] : backends) {
    RuntimeOptions options;
    options.par_backend = backend;
    if (backend == ParBackend::kThreads) {
      options.threads = 4;
    }

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Sigmoid " << name << " time: " << duration.count() << " ms"
              << std::endl;

    auto current_result = *out[0].as<int>();
    if (first) {
      reference_result = current_result;
      first = false;
      for (size_t i = 0; i < 100; i++) {
        EXPECT_EQ(current_result[i], 0)
            << "Invalid sigmoid result at index " << i;
      }
    } else {
      for (size_t i = 0; i < reference_result.size(); i++) {
        EXPECT_EQ(current_result[i], reference_result[i])
            << "Mismatch with " << name << " at index " << i;
      }
    }
  }
}

TEST(ewlayer, parallel_for_direct) {
  const int SIZE = 2000;
  std::vector<int> matrix1(SIZE * SIZE);
  std::vector<int> matrix2(SIZE * SIZE);
  std::vector<int> result(SIZE * SIZE);

  for (int i = 0; i < SIZE * SIZE; ++i) {
    matrix1[i] = 1;
    matrix2[i] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kSeq);

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kThreads);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kTbb);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kOmp);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }
}

TEST(ewlayer, parallel_for_notmatrix) {
  const int SIZE = 3000;
  std::vector<int> matrix1(SIZE * SIZE);
  std::vector<int> result(SIZE * SIZE);

  for (int i = 0; i < SIZE * SIZE; ++i) {
    matrix1[i] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kSeq);

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kThreads);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kTbb);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kOmp);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }
}

struct EWLayerTestParams {
  std::string activation_type;
  float alpha;
  float beta;
  std::vector<float> input;
  Shape input_shape;
  std::vector<float> expected_output;
  std::string description;
};

class EWLayerParametrizedTest
    : public BaseTestFixture,
      public ::testing::WithParamInterface<
          std::tuple<EWLayerTestParams, RuntimeOptions>> {};

TEST_P(EWLayerParametrizedTest, test_activation_with_different_backends) {
  auto [params, runtime_options] = GetParam();

  EWLayer layer(params.activation_type, params.alpha, params.beta);

  Tensor input = make_tensor<float>(params.input, params.input_shape);
  Tensor output;

  std::vector<Tensor> inputs{input};
  std::vector<Tensor> outputs{output};

  layer.run(inputs, outputs, runtime_options);

  auto output_data = *outputs[0].as<float>();
  expectVectorsNear(output_data, params.expected_output, 1e-4f);
}

INSTANTIATE_TEST_SUITE_P(
    EWLayerTests, EWLayerParametrizedTest,
    ::testing::Combine(
        ::testing::Values(
            EWLayerTestParams{"relu", 1.0f, 0.0f,
                              BaseTestFixture::activationTestData(), Shape{7},
                              BaseTestFixture::reluExpected(), "ReLU"},
            EWLayerTestParams{"sigmoid", 1.0f, 0.0f,
                              BaseTestFixture::activationTestData(), Shape{7},
                              BaseTestFixture::sigmoidExpected(), "Sigmoid"},
            EWLayerTestParams{"linear",
                              2.0f,
                              3.0f,
                              {1.0f, -1.0f, 2.0f, -2.0f, 0.0f},
                              Shape{5},
                              {5.0f, 1.0f, 7.0f, -1.0f, 3.0f},
                              "Linear_2x_plus_3"},
            EWLayerTestParams{"linear", 1.0f, 0.0f,
                              BaseTestFixture::basic1DData(),
                              BaseTestFixture::basic1DShape(),
                              BaseTestFixture::basic1DData(), "Linear_x"},
            EWLayerTestParams{
                "tanh",
                1.0f,
                0.0f,
                {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f},
                Shape{5},
                {std::tanh(-2.0f), std::tanh(-1.0f), std::tanh(0.0f),
                 std::tanh(1.0f), std::tanh(2.0f)},
                "Tanh"},
            EWLayerTestParams{
                "relu", 1.0f, 0.0f, BaseTestFixture::ascending1DData(),
                BaseTestFixture::ascending1DShape(),
                BaseTestFixture::ascending1DData(), "ReLU_Ascending"},
            EWLayerTestParams{
                "relu",
                1.0f,
                0.0f,
                BaseTestFixture::mixed1DData(),
                BaseTestFixture::ascending1DShape(),
                {0.0f, 0.0f, 0.0f, 2.0f, 4.0f, 0.0f, 3.0f, 1.0f, 0.0f, 5.0f},
                "ReLU_Mixed"},
            EWLayerTestParams{"relu", 1.0f, 0.0f, std::vector<float>(10, 0.0f),
                              Shape{10}, std::vector<float>(10, 0.0f),
                              "ReLU_All_Zeros"},
            EWLayerTestParams{"relu", 1.0f, 0.0f,
                              BaseTestFixture::basic2DData4x4(),
                              BaseTestFixture::basic2DShape4x4(),
                              BaseTestFixture::basic2DData4x4(), "ReLU_2D"},
            EWLayerTestParams{
                "relu",
                1.0f,
                0.0f,
                {9.0f, -8.0f, 7.0f, -6.0f, -5.0f, 4.0f, -3.0f, 2.0f, 2.0f,
                 -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f, -9.0f},
                Shape{4, 4},
                {9.0f, 0.0f, 7.0f, 0.0f, 0.0f, 4.0f, 0.0f, 2.0f, 2.0f, 0.0f,
                 4.0f, 0.0f, 6.0f, 0.0f, 8.0f, 0.0f},
                "ReLU_2D_Mixed"}),
        ::testing::Values(BaseTestFixture::setTBBOptions(),
                          BaseTestFixture::setSeqOptions(),
                          BaseTestFixture::setOmpOptions(),
                          BaseTestFixture::setKokkosOptions(),
                          BaseTestFixture::setSTLOptions())),
    [](const ::testing::TestParamInfo<
        std::tuple<EWLayerTestParams, RuntimeOptions>>& info) {
      const auto& params = std::get<0>(info.param);
      const auto& options = std::get<1>(info.param);

      std::string name = params.description + "_";
      if (options.par_backend == ParBackend::kTbb) {
        name += "TBB";
      } else if (options.par_backend == ParBackend::kThreads) {
        name += "STL";
      } else if (options.par_backend == ParBackend::kOmp) {
        name += "OMP";
      } else if (options.par_backend == ParBackend::kKokkos) {
        name += "Kokkos";
      } else {
        name += "Seq";
      }

      std::replace(name.begin(), name.end(), ' ', '_');
      std::replace(name.begin(), name.end(), '-', '_');
      std::replace(name.begin(), name.end(), '.', '_');
      return name;
    });
