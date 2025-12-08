#include <algorithm>
#include <vector>

#include "graph/graph.hpp"
#include "layers/Layer.hpp"

namespace it_lab_ai {
std::vector<std::vector<int>> find_subgraphs(const Graph& graph,
                                             const Graph& subgraph);
bool has_edge(const Graph& graph, int id_from, int id_to);
bool is_root(const Graph& graph, int id);
bool is_leaf(const Graph& graph, int id);
bool run_search(const Graph& graph, const Graph& subgraph,
                std::vector<int>& assignments,
                std::vector<std::vector<int>>& results);
}  // namespace it_lab_ai
