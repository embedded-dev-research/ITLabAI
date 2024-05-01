#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/Tensor.hpp"

using namespace itlab_2023;

struct TestClass {
 public:
  TestClass(float a) : b_(a) {}
  char get_a() const { return a_; }
  float get_b() const { return b_; }

 private:
  char a_{'1'};
  float b_;
};

TEST(Tensor, can_create_int_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  ASSERT_NO_THROW(make_tensor<int>(vals_tensor, sh));
}

TEST(Tensor, can_create_float_tensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  ASSERT_NO_THROW(make_tensor<float>(vals_tensor, sh));
}

TEST(Tensor, can_get_tensor_type) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  const Tensor tensor = make_tensor(vals_tensor, sh);
  EXPECT_EQ(tensor.get_type(), Type::kFloat);
}

TEST(Tensor, can_iterate_through_tensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  size_t i = 0;
  for (auto it = vals_tensor.begin(); it != vals_tensor.end(); it++) {
    EXPECT_NEAR(*it, vals_tensor[i], 1e-5);
    i++;
  }
}

TEST(Tensor, check_get_positive_integer_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  EXPECT_EQ(t.get<int>({1, 1}), 6);
}

TEST(Tensor, check_get_negative_integer_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  EXPECT_EQ(t.get<int>({0, 2}), -2);
}

TEST(Tensor, check_get_positive_float_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh);
  EXPECT_NEAR(t.get<float>({1, 2}), 3.0, 1e-5);
}

TEST(Tensor, check_get_negative_float_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh);
  EXPECT_NEAR(t.get<float>({0, 1}), -0.2, 1e-5);
}

TEST(Tensor, check_get_float_element_from_const_tensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  const Tensor t = make_tensor<float>(vals_tensor, sh);
  EXPECT_NEAR(t.get<float>({0, 1}), -0.2, 1e-5);
}

TEST(Tensor, check_get_operation_with_out_of_range_coordinates) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  ASSERT_ANY_THROW(t.get<int>({2, 2}));
}

TEST(Tensor, check_set_operation_with_out_of_range_coordinates) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  ASSERT_ANY_THROW(t.set<int>({2, 2}, 6));
}

TEST(Tensor, check_set_integer_element_in_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  t.set<int>({1, 1}, 99);
  EXPECT_EQ(t.get<int>({1, 1}), 99);
}

TEST(Tensor, check_set_float_element_in_tensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh);
  t.set<float>({1, 1}, 99.2F);
  EXPECT_NEAR(t.get<float>({1, 1}), 99.2F, 1e-5);
}

TEST(Tensor,
     cannot_get_element_using_the_get_method_with_an_inappropriate_type) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  ASSERT_ANY_THROW(t.get<float>({0, 0}));
}

TEST(Tensor, cannot_create_tensor_based_on_an_unsuitable_vector_of_values) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3, 9, 1};
  ASSERT_ANY_THROW(make_tensor<int>(vals_tensor, sh););
}

TEST(Tensor, cannot_create_a_tensor_with_an_invalid_type) {
  Shape sh({2, 3});
  std::vector<TestClass> vals_tensor = {4.0F, 0.3F, -2.1F, 1.9F, 6.3F, -3.0F};
  ASSERT_ANY_THROW(make_tensor<TestClass>(vals_tensor, sh););
}

TEST(Tensor, can_interpret_as_test) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  const Tensor t = make_tensor<float>(vals_tensor, sh);
  std::vector<float> tmp_tensor = *t.as<float>();
  for (size_t i = 0; i < sh.count(); i++) {
    EXPECT_NEAR(tmp_tensor[i], vals_tensor[i], 1e-5);
  }
}
