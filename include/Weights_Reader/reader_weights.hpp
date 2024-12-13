#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "layers/Tensor.hpp"

using json = nlohmann::json;
using namespace itlab_2023;

json read_json(const std::string& filename);
void extract_values_without_bias(const json& j, std::vector<float>& values);
void extract_values_from_json(const json& j, std::vector<float>& values);
void parse_json_shape(const json& j, std::vector<size_t>& shape, size_t dim);
Tensor create_tensor_from_json(const json& j, Type type);
void extract_bias_from_json(const json& j, std::vector<float>& bias);
