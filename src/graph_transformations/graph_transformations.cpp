#include "graph/graph_transformations.hpp"

namespace it_lab_ai {

std::vector<int> find_subgraphs(const Graph& graph, const Graph& subgraph) {
  // requirements for subgraph:
  // 1 input, 1 output
  // requirements for graph:
  // can't be connected with subgraph from _outside_, except input and output
  std::vector<int> result;
  for (int i = 0; i < graph.getLayersCount(); i++) {
    bool temp = check_child(graph, subgraph, i, 0); // recursion starts
    if (temp) {
      result.push_back(i);
    }
  }
  return result;
}

bool layer_conditions(const Layer& layer, const Layer& layer_sub) {
  return layer.getName() == layer_sub.getName();
}

bool check_child(const Graph& graph, const Graph& subgraph, int i, int iter) {
  int amount_connected1 =
      (i < graph.getLayersCount() - 1)
          ? (graph.getVertexValue(i + 1) - graph.getVertexValue(i))
          : 0;
  int amount_connected2 =
      (iter < subgraph.getLayersCount() - 1)
          ? (subgraph.getVertexValue(iter + 1) - subgraph.getVertexValue(iter))
          : 0;
  if ((amount_connected2 != 0 && amount_connected1 != amount_connected2) ||
      !layer_conditions(*graph.getLayerFromID(i),
                        *subgraph.getLayerFromID(iter))) {
    return false;
  }
  if (amount_connected2 != 0) {
    using id_name = std::pair<int, LayerType>;
    std::vector<id_name> orderA;
    std::vector<id_name> orderB;
    for (int j = 0; j < amount_connected1; j++) {
      orderA.push_back(id_name(
          graph.getEdgeValue(graph.getVertexValue(i) + j),
          graph.getLayerFromID(graph.getEdgeValue(graph.getVertexValue(i) + j))
              ->getName()));
      orderB.push_back(id_name(
          subgraph.getEdgeValue(subgraph.getVertexValue(iter) + j),
          subgraph
              .getLayerFromID(
                  subgraph.getEdgeValue(subgraph.getVertexValue(iter) + j))
              ->getName()));
    }
    std::sort(orderA.begin(), orderA.end(),
              [&](id_name a1, id_name a2) { return a1.second < a2.second; });
    // ^ interested in LayerType order to prevent any shuffling for childs
    for (int j = 0; j < amount_connected1; j++) {
      if (graph.getInputsSize(j) != subgraph.getInputsSize(iter)) {
        return false;
      }
      bool temp =
          check_child(graph, subgraph, orderA[j].first, orderB[j].second);
      if (!temp) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace it_lab_ai