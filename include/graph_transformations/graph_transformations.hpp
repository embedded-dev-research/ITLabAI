#include <vector>

#include "graph/graph.hpp"
#include "layers/Layer.hpp"

namespace it_lab_ai {
std::vector<int> find_subgraphs(const Graph& graph, const Graph& subgraph);
bool layer_conditions(const Layer& layer, const Layer& layer_sub);
bool check_child(const Graph& graph, const Graph& subgraph, int i, int iter, int recursionDepth);
}  // namespace it_lab_ai
