#include <cstdint>
#include "gtest/gtest.h"
#include "layers/Tensor.hpp"
#include <vector>

TEST(Tensor, can_create_int_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  ASSERT_NO_THROW( make_tensor<int>(vals_tensor, sh) );
}

TEST(Tensor, can_create_double_tensor) {
  Shape sh({2, 3});
  std::vector<double> vals_tensor = {4.5, -0.2, 2.1, -1.7, -6.9, 3.0};
  ASSERT_NO_THROW( make_tensor<double>(vals_tensor, sh) );
}

TEST(Tensor, check_get_positive_integer_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  EXPECT_EQ( t.get<int>({1, 1}), 6 );
}

TEST(Tensor, check_get_negative_integer_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  EXPECT_EQ( t.get<int>({0, 2}), -2 );
}

TEST(Tensor, check_get_positive_double_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<double> vals_tensor = {4.5, -0.2, 2.1, -1.7, -6.9, 3.0};
  Tensor t = make_tensor<double>(vals_tensor, sh);
  EXPECT_NEAR( t.get<double>({1, 2}), 3.0, 1e-5 );
}

TEST(Tensor, check_get_negative_double_element_from_tensor) {
  Shape sh({2, 3});
  std::vector<double> vals_tensor = {4.5, -0.2, 2.1, -1.7, -6.9, 3.0};
  Tensor t = make_tensor<double>(vals_tensor, sh);
  EXPECT_NEAR( t.get<double>({0, 1}), -0.2, 1e-5 );
}

TEST(Tensor, check_get_operation_with_out_of_range_coordinates) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  ASSERT_ANY_THROW( t.get<int>({2, 2}) );
}

TEST(Tensor, cannot_get_element_using_the_get_method_with_an_inappropriate_type) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  ASSERT_ANY_THROW( t.get<double>({0, 0}) );
}

TEST(Tensor, cannot_create_tensor_tensor_based_on_an_unsuitable_vector_of_values) {
  Shape sh({2, 3});
  std::vector<int> vals_tensor = {4, 0, -2, 1, 6, -3, 9, 1};
  ASSERT_ANY_THROW( make_tensor<int>(vals_tensor, sh); );
}


TEST(Tensor, cannot_create_a_tensor_with_an_invalid_type) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.0f, 0.3f, -2.1f, 1.9f, 6.3f, -3.0f};
  ASSERT_ANY_THROW( make_tensor<float>(vals_tensor, sh); );
}