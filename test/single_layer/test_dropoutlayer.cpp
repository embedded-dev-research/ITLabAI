#include <numeric>
#include <vector>

#include "gtest/gtest.h"
#include "layers/DropOutLayer.hpp"

using namespace it_lab_ai;

TEST(DropOutLayer, IncompatibleInput) {
  DropOutLayer layer(1);
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  std::vector<Tensor> in{input, input};
  std::vector<Tensor> out{output};
  ASSERT_ANY_THROW(layer.run(in, out));
}

TEST(DropOutLayer, dropoutlayer_int) {
  DropOutLayer layer(1);
  Shape sh({2, 2});
  Tensor input = make_tensor<int>({1, -1, 2, -2}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<int> vec = *out[0].as<int>();
  EXPECT_EQ(vec[0], 0);
  EXPECT_EQ(vec[1], 0);
  EXPECT_EQ(vec[2], 0);
  EXPECT_EQ(vec[3], 0);
}

TEST(DropOutLayer, dropoutlayer_float) {
  DropOutLayer layer(0);
  Shape sh({2, 2});
  Tensor input = make_tensor<float>({1.0F, -1.0F, 2.0F, -2.0F}, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> vec = *out[0].as<float>();
  EXPECT_NEAR(vec[0], 1, 1e-5);
  EXPECT_NEAR(vec[1], -1, 1e-5);
  EXPECT_NEAR(vec[2], 2, 1e-5);
  EXPECT_NEAR(vec[3], -2, 1e-5);
}

TEST(DropOutLayer, dropoutlayer_float_50proc) {
  DropOutLayer layer(0.5);
  Shape sh({10, 10});
  std::vector<float> a(100, static_cast<float>(0.01));
  Tensor input = make_tensor<float>(a, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> vec = *out[0].as<float>();
  EXPECT_NEAR(std::accumulate(vec.begin(), vec.end(), 0.0F), 0.5, 0.2);
}

TEST(DropOutLayer, dropoutlayer_float_30proc) {
  DropOutLayer layer(0.3);
  Shape sh({10, 10});
  std::vector<float> a(100, static_cast<float>(0.01));
  Tensor input = make_tensor<float>(a, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> vec = *out[0].as<float>();
  EXPECT_NEAR(std::accumulate(vec.begin(), vec.end(), 0.0F), 0.7, 0.2);
}

TEST(DropOutLayer, dropoutlayer_float_70proc) {
  DropOutLayer layer(0.7);
  Shape sh({10, 10});
  std::vector<float> a(100, static_cast<float>(0.01));
  Tensor input = make_tensor<float>(a, sh);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};
  layer.run(in, out);
  std::vector<float> vec = *out[0].as<float>();
  EXPECT_NEAR(std::accumulate(vec.begin(), vec.end(), 0.0F), 0.3, 0.2);
}
