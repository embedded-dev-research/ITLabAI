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

TEST(Tensor, can_create_float_tensor_with_bias) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  std::vector<float> bias = {0.5F, 1.5F, -1.0F};
  ASSERT_NO_THROW(make_tensor<float>(vals_tensor, sh, bias));
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
  ASSERT_ANY_THROW(make_tensor<int>(vals_tensor, sh));
}

TEST(Tensor, cannot_create_a_tensor_with_an_invalid_type) {
  Shape sh({2, 3});
  std::vector<TestClass> vals_tensor = {4.0F, 0.3F, -2.1F, 1.9F, 6.3F, -3.0F};
  ASSERT_ANY_THROW(make_tensor<TestClass>(vals_tensor, sh));
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

TEST(Tensor, can_set_and_get_bias) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  std::vector<float> bias = {0.5F, 1.5F, -1.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh);
  t.set_bias(bias);
  const std::vector<float>& retrieved_bias = t.get_bias();
  ASSERT_EQ(retrieved_bias.size(), bias.size());
  for (size_t i = 0; i < bias.size(); ++i) {
    EXPECT_NEAR(retrieved_bias[i], bias[i], 1e-5);
  }
}

TEST(Tensor, can_create_tensor_with_bias_and_get_bias) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  std::vector<float> bias = {0.5F, 1.5F, -1.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh, bias);
  const std::vector<float>& retrieved_bias = t.get_bias();
  ASSERT_EQ(retrieved_bias.size(), bias.size());
  for (size_t i = 0; i < bias.size(); ++i) {
    EXPECT_NEAR(retrieved_bias[i], bias[i], 1e-5);
  }
}

TEST(Tensor, check_set_bias_after_creation) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh);
  std::vector<float> new_bias = {1.0F, 0.5F, -0.5F};
  t.set_bias(new_bias);
  const std::vector<float>& retrieved_bias = t.get_bias();
  ASSERT_EQ(retrieved_bias.size(), new_bias.size());
  for (size_t i = 0; i < new_bias.size(); ++i) {
    EXPECT_NEAR(retrieved_bias[i], new_bias[i], 1e-5);
  }
}

TEST(Tensor, cannot_set_bias_with_incorrect_size) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  Tensor t = make_tensor<float>(vals_tensor, sh);
  std::vector<float> incorrect_bias = {0.5F, 1.5F};
  ASSERT_ANY_THROW(t.set_bias(incorrect_bias));
}

TEST(Tensor, can_create_multidimensional_tensor) {
  Shape sh({2, 3, 2});  // 3D tensor shape
  std::vector<int> vals_tensor = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ASSERT_NO_THROW(make_tensor<int>(vals_tensor, sh));
}

TEST(Tensor, check_get_element_from_multidimensional_tensor) {
  Shape sh({2, 3, 2});  // 3D tensor shape
  std::vector<int> vals_tensor = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  EXPECT_EQ(t.get<int>({1, 2, 1}), 12);
}

TEST(Tensor, cannot_get_element_with_invalid_coordinates) {
  Shape sh({2, 3, 2});  // 3D tensor shape
  std::vector<int> vals_tensor = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  Tensor t = make_tensor<int>(vals_tensor, sh);
  ASSERT_ANY_THROW(t.get<int>({2, 3, 1}));
}
TEST(Tensor, cannot_create_tensor_with_incorrect_shape) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {1.0F, 2.0F, 3.0F, 4.0F,
                                    5.0F};  // Incorrect size
  ASSERT_ANY_THROW(make_tensor<float>(vals_tensor, sh));
}

TEST(Tensor, cannot_create_tensor_with_unknown_type) {
  std::vector<char> vals_tensor = {'a', 'b', 'c'};
  Shape sh({3});
  ASSERT_ANY_THROW(make_tensor<char>(vals_tensor, sh));
}

TEST(TensorTest, EmptyTensor) {
  Tensor tensor;
  EXPECT_TRUE(tensor.empty());
}

TEST(TensorTest, NonEmptyTensor) {
  Shape sh({2, 3});
  std::vector<float> vals_tensor = {4.5F, -0.2F, 2.1F, -1.7F, -6.9F, 3.0F};
  const Tensor tensor = make_tensor(vals_tensor, sh);

  EXPECT_FALSE(tensor.empty());
}

TEST(TensorTest, TensorWithBias) {
  Shape shape({2, 3});
  std::vector<float> values = {1, 2, 3, 4, 5, 6};
  std::vector<float> bias = {1, 2, 3};
  Tensor tensor = make_tensor(values, shape, bias);

  EXPECT_EQ(tensor.get_bias().size(), 3);
  EXPECT_EQ(tensor.get_bias()[0], 1);
  EXPECT_EQ(tensor.get_bias()[1], 2);
  EXPECT_EQ(tensor.get_bias()[2], 3);
}

TEST(TensorTest, CorrectBiasSize) {
  Shape shape({2, 3});
  std::vector<float> values = {1, 2, 3, 4, 5, 6};
  std::vector<float> correct_bias = {1, 2, 3};
  EXPECT_NO_THROW(Tensor tensor = make_tensor(values, shape, correct_bias));
}

TEST(TensorTest, IncorrectBiasSize) {
  Shape shape({2, 3});
  std::vector<float> values = {1, 2, 3, 4, 5, 6};
  std::vector<float> incorrect_bias = {1.0f, 2.0f};
  EXPECT_THROW(Tensor tensor = make_tensor(values, shape, incorrect_bias),
               std::invalid_argument);
}
