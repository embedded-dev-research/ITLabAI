#include <algorithm>
#include <vector>

#include "graph/graph.hpp"
#include "layers/Layer.hpp"
#include "layers/EWLayer.hpp"

namespace it_lab_ai {
std::vector<std::vector<int>> find_subgraphs(const Graph& graph,
                                             const Graph& subgraph);
bool has_edge(const Graph& graph, int id_from, int id_to);
bool is_root(const Graph& graph, int id);
bool is_leaf(const Graph& graph, int id);
bool run_search(const Graph& graph, const Graph& subgraph,
                std::vector<int>& assignments,
                std::vector<std::vector<int>>& results);

void change_ids(std::vector<std::vector<int>>& vec, int id);
bool does_intersect(const std::vector<int>& vec1, const std::vector<int>& vec2);
Graph changed_subgraphs(const Graph& graph, const Graph& subgraph_from);
}  // namespace it_lab_ai
