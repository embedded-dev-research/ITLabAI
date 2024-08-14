#include <gtest/gtest.h>

#include <fstream>

#include "Weights_Reader/reader_weights.hpp"

std::string get_test_data_path(const std::string& filename) {
  return std::string(TEST_DATA_PATH) + "/" + filename;
}

TEST(ReaderWeightsTest, ReadJsonFailed) {
  std::string filename = get_test_data_path("error.json");
  EXPECT_THROW(read_json(filename), std::runtime_error);
}

TEST(ReaderWeightsTest, ReadEmptyJson) {
  std::string filename = get_test_data_path("empty0.json");
  EXPECT_THROW(read_json(filename), std::runtime_error);
}

TEST(ReaderWeightsTest, ReadJson_UnexpectedType) {
  std::string filename = get_test_data_path("empty0.json");
  EXPECT_THROW(read_json(filename), std::runtime_error);
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

TEST(ExtractValuesFromJsonTests, HandlesFlatArray) {
  json j = json::array({1.0, 2.0, 3.0, 4.0});
  std::vector<float> values;
  extract_values_from_json(j, values);
  std::vector<float> expected = {1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(values, expected);
}

TEST(ExtractValuesFromJsonTests, HandlesNestedArray) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  std::vector<float> values;
  extract_values_from_json(j, values);
  std::vector<float> expected = {1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(values, expected);
}

TEST(ParseJsonShapeTests, HandlesEmptyArray) {
  json j = json::array({});
  std::vector<size_t> shape;
  parse_json_shape(j, shape);
  std::vector<size_t> expected = {0};
  EXPECT_EQ(shape, expected);
}

TEST(ParseJsonShapeTests, HandlesSimpleArray) {
  json j = json::array({{1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}});
  std::vector<size_t> shape;
  parse_json_shape(j, shape);
  std::vector<size_t> expected = {3};
  EXPECT_EQ(shape, expected);
}

TEST(ParseJsonShapeTests, HandlesNestedArray) {
  json j = json::array({{{1.0, 2.0}, {3.0, 4.0}}, {3.0, 4.0}});
  std::vector<size_t> shape;
  parse_json_shape(j, shape);
  std::vector<size_t> expected = {2, 2};
  EXPECT_EQ(shape, expected);
}

TEST(ExtractValuesWithoutBiasTest, HandlesCaseWithoutBias) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  std::vector<float> values;
  extract_values_without_bias(j, values);
  std::vector<float> expected = {1.0, 2.0};
  EXPECT_EQ(values, expected);
}

TEST(ExtractValuesWithoutBiasTest, HandlesEmptyJson) {
  json j = json::array({});
  std::vector<float> values;
  extract_values_without_bias(j, values);
  EXPECT_TRUE(values.empty());
}

TEST(ExtractValuesWithoutBiasTest, HandlesComplexNestedCase) {
  json j = json::array({{{1.0, 2.0}, {3.0, 4.0}}, {5.0, 6.0}});
  std::vector<float> values;
  extract_values_without_bias(j, values);
  std::vector<float> expected = {1.0, 2.0, 3.0, 4.0};
  EXPECT_EQ(values, expected);
}

TEST(ExtractBiasFromJsonTests, extract_bias) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  std::vector<float> values;
  extract_bias_from_json(j, values);
  std::vector<float> expected = {3.0, 4.0};
  EXPECT_EQ(values, expected);
}

TEST(ExtractBiasFromJsonTests, extract_bias_error_array) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  std::vector<float> values;
  extract_bias_from_json(j, values);
  std::vector<float> expected = {3.0, 4.0};
  EXPECT_EQ(values, expected);
}

TEST(CreateTensorFromJsonTest, SimpleTensor) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  EXPECT_NO_THROW(Tensor tensor = create_tensor_from_json(j, Type::kFloat););
}

TEST(CreateTensorFromJsonTest, SimpleTensorCheckBias) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  Tensor tensor = create_tensor_from_json(j, Type::kFloat);

  EXPECT_EQ(tensor.get_bias().size(), 2);
  EXPECT_EQ(tensor.get_bias()[0], 3.0);
  EXPECT_EQ(tensor.get_bias()[1], 4.0);
}

TEST(CreateTensorFromJsonTest, SimpleTensorCheckWeights) {
  json j = json::array({{1.0, 2.0}, {3.0, 4.0}});
  Tensor tensor = create_tensor_from_json(j, Type::kFloat);

  EXPECT_EQ(tensor.get<float>({1}), 2.0);
}

TEST(CreateTensorFromJsonTest, SimpleTensorCheckNoBias) {
  json j = json::array({{1.0, 2.0}});
  ASSERT_ANY_THROW(Tensor tensor = create_tensor_from_json(j, Type::kFloat););
}

TEST(CreateTensorFromJsonTest, EmptyShape) {
  json j = json::array({});
  std::vector<size_t> shape;
  parse_json_shape(j, shape);
  std::vector<size_t> expected = {0};
  EXPECT_EQ(shape, expected);
}

TEST(CreateTensorFromJsonTest, CheckIntTensor) {
  json j = json::array({{1, 2}, {3, 7}});
  ASSERT_ANY_THROW(Tensor tensor = create_tensor_from_json(j, Type::kInt););
}

TEST(ParseJsonShapeTests, HandlesNonArrayJson) {
  json j = "not_an_array";
  std::vector<size_t> shape;

  parse_json_shape(j, shape);

  std::vector<size_t> expected = {0};
  EXPECT_EQ(shape, expected);
}
