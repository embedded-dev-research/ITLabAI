#include "graph_transformations/graph_transformations.hpp"

namespace it_lab_ai {

bool layer_conditions(const Layer& layer, const Layer& layer_sub) {
  return layer.getName() == layer_sub.getName();
}

std::vector<std::vector<int>> find_subgraphs(const Graph& graph,
                                             const Graph& subgraph) {
  // requirements for subgraph:
  // one or multiple inputs, one or multiple outputs
  // requirements for graph:
  // can't be connected from outside, except IO for input and O for output
  std::vector<int> assignments;  // cur assumption for graph
  std::vector<std::vector<int>> results;
  run_search(graph, subgraph, assignments, results);
  return results;
}

bool has_edge(const Graph& graph, int id_from, int id_to) {
  for (int i = graph.getVertexValue(id_from);
       i < graph.getVertexValue(id_from + 1); i++) {
    if (graph.getEdgeValue(i) == id_to) {
      return true;
    }
  }
  return false;
}

bool is_root(const Graph& graph, int id) {
  return graph.getInputsSize(id) == 0;
}

bool is_leaf(const Graph& graph, int id) {
  return graph.getVertexValue(id + 1) - graph.getVertexValue(id) == 0;
}

bool run_search(const Graph& graph, const Graph& subgraph,
                std::vector<int>& assignments,
                std::vector<std::vector<int>>& results) {
  size_t cur_size = assignments.size();
  for (int prev_id = 0; prev_id < subgraph.getLayersCount(); prev_id++) {
    int amount_connected_s =
        subgraph.getVertexValue(prev_id + 1) - subgraph.getVertexValue(prev_id);
    for (int j = 0; j < amount_connected_s; j++) {
      int next_id = subgraph.getEdgeValue(subgraph.getVertexValue(prev_id) + j);
      if (prev_id < static_cast<int>(cur_size) &&
          next_id < static_cast<int>(cur_size)) {
        if (!has_edge(graph, assignments[prev_id], assignments[next_id])) {
          return false;
        }
        std::vector<int> ids = {prev_id, next_id};
        for (int k = 0; k < 2; k++) {
          if (!layer_conditions(subgraph.getLayerFromID(ids[k]),
                                graph.getLayerFromID(assignments[ids[k]]))) {
            return false;
          }
          // input node shouldn't be checked for it's inputs
          if (!is_root(subgraph, ids[k]) &&
              subgraph.getInputsSize(ids[k]) !=
                  graph.getInputsSize(assignments[ids[k]])) {
            return false;
          }
          // input & output node shouldn't be checked for it's outputs
          if (!is_leaf(subgraph, ids[k]) && !is_root(subgraph, ids[k])) {
            int amount_connected_s1 = subgraph.getVertexValue(ids[k] + 1) -
                                      subgraph.getVertexValue(ids[k]);
            int amount_connected_1 =
                graph.getVertexValue(assignments[ids[k]] + 1) -
                graph.getVertexValue(assignments[ids[k]]);
            if (amount_connected_1 != amount_connected_s1) {
              return false;
            }
          }
        }
      }
    }
  }

  // assumption is good -> return true
  if (static_cast<int>(cur_size) == subgraph.getLayersCount()) {
    return true;
  }

  // add new nodes for assumption and try recursion
  for (int id = 0; id < graph.getLayersCount(); id++) {
    auto it = std::find(assignments.begin(), assignments.end(), id);
    if (it == assignments.end()) {
      assignments.push_back(id);
      if (run_search(graph, subgraph, assignments, results)) {
        results.emplace_back(assignments);
      }
      assignments.pop_back();
    }
  }
  return false;
}

}  // namespace it_lab_ai
