#include <gtest/gtest.h>

#include <fstream>

#include "Weights_Reader/reader_weights.hpp"

std::string get_test_data_path(const std::string& filename) {
  return std::string(TEST_DATA_PATH) + "/" + filename;
}

TEST(ReaderWeightsTest, ReadJsonValidLargeFile) {
  std::string filename = get_test_data_path("valid.json");
  json j = read_json(filename);
  EXPECT_TRUE(j.contains("layer1"));
  EXPECT_TRUE(j.contains("layer2"));
  EXPECT_TRUE(j.contains("layer3"));
  EXPECT_TRUE(j.contains("layer4"));
}
TEST(ReaderWeightsTest, ReadJsonEmptyFile) {
  std::string filename = get_test_data_path("empty.json");
  json j = read_json(filename);
  EXPECT_TRUE(j.empty());
}

TEST(ReaderWeightsTest, ReadJsonInvalidFile) {
  std::string filename = get_test_data_path("invalid-[.json");
  std::string filename1 = get_test_data_path("invalid-_.json");
  std::string filename2 = get_test_data_path("invalid_number.json");
  std::string filename3 = get_test_data_path("invalid-}.json");
  std::string filename4 = get_test_data_path("invalid-}}.json");

  EXPECT_THROW(read_json(filename), std::runtime_error);
  EXPECT_THROW(read_json(filename1), std::runtime_error);
  EXPECT_THROW(read_json(filename2), std::runtime_error);
  EXPECT_THROW(read_json(filename3), std::runtime_error);
  EXPECT_THROW(read_json(filename4), std::runtime_error);
}

TEST(ReaderWeightsTest, ExtractValuesFromJson) {
  json j = json::array({1.0, 2.0, 3.0});
  std::vector<float> values;
  extract_values_from_json(j, values);
  ASSERT_EQ(values.size(), 3);
  EXPECT_FLOAT_EQ(values[0], 1.0);
  EXPECT_FLOAT_EQ(values[1], 2.0);
  EXPECT_FLOAT_EQ(values[2], 3.0);
}

TEST(ReaderWeightsTest, ExtractValuesFromNestedJson) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  std::vector<float> values;
  extract_values_from_json(j, values);
  ASSERT_EQ(values.size(), 4);
  EXPECT_FLOAT_EQ(values[0], 1.0);
  EXPECT_FLOAT_EQ(values[1], 2.0);
  EXPECT_FLOAT_EQ(values[2], 3.0);
  EXPECT_FLOAT_EQ(values[3], 4.0);
}

TEST(ReaderWeightsTest, ExtractValuesFromJsonWithInvalidType) {
  json j = "string";
  std::vector<float> values;
  EXPECT_THROW(extract_values_from_json(j, values), std::runtime_error);
}

TEST(ReaderWeightsTest, CreateTensorFromJsonInvalidType) {
  json j = json::array({1.0, 2.0, 3.0});
  EXPECT_THROW(create_tensor_from_json(j, static_cast<Type>(-1)),
               std::invalid_argument);
}

TEST(ReaderWeightsTest, CreateTensorFromJsonInvalidJson) {
  json j = "string";
  EXPECT_THROW(create_tensor_from_json(j, Type::kFloat), std::runtime_error);
}

TEST(TensorFromJson, can_create_tensor_from_valid_json) {
  std::string filename = get_test_data_path("valid.json");
  json j = read_json(filename);
    ASSERT_NO_THROW({
      Tensor tensor1 = create_tensor_from_json(j["layer1"], Type::kFloat);
      EXPECT_EQ(tensor1.get_shape().dims(), 1);
      EXPECT_EQ(tensor1.get_shape()[0], 5);
    });

    ASSERT_NO_THROW({
      Tensor tensor2 = create_tensor_from_json(j["layer2"], Type::kFloat);
      EXPECT_EQ(tensor2.get_shape().dims(), 2);
      EXPECT_EQ(tensor2.get_shape()[0], 2);
      EXPECT_EQ(tensor2.get_shape()[1], 5);
    });

    ASSERT_NO_THROW({
      Tensor tensor3 =
          create_tensor_from_json(j["layer3"]["sub_layer1"], Type::kFloat);
      EXPECT_EQ(tensor3.get_shape().dims(), 1);
      EXPECT_EQ(tensor3.get_shape()[0], 3);
    });

    ASSERT_NO_THROW({
      Tensor tensor4 = create_tensor_from_json(j["layer4"], Type::kFloat);
      EXPECT_EQ(tensor4.get_shape().dims(), 2);
      EXPECT_EQ(tensor4.get_shape()[0], 2);
      EXPECT_EQ(tensor4.get_shape()[1], 2);
    });
}