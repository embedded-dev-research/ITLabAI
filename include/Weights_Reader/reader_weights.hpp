#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "layers/Tensor.hpp"

using json = nlohmann::json;
using namespace itlab_2023;

json read_json(const std::string& filename);

void extract_values_from_json(const json& j, std::vector<float>& values);

Tensor create_tensor_from_json(const json& j, Type type);

