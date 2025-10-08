#include "graph_transformations/graph_transformations.hpp"

namespace it_lab_ai {

std::vector<int> find_subgraphs(const Graph& graph, const Graph& subgraph) {
  // requirements for subgraph:
  // 1 input, 1 output
  // requirements for graph:
  // can't be connected with subgraph from _outside_, except input and output
  std::vector<int> result;
  for (int i = 0; i < graph.getLayersCount(); i++) {
    bool temp = check_child(graph, subgraph, i, 0, 0);  // recursion starts
    if (temp) {
      result.push_back(i);
    }
  }
  return result;
}

bool layer_conditions(const Layer& layer, const Layer& layer_sub) {
  std::cerr << "Comparing type " << static_cast<int>(layer.getName()) << " and "
            << static_cast<int>(layer_sub.getName()) << std::endl;
  return layer.getName() == layer_sub.getName();
}

// void erase_inequality_for_first(
//     std::vector<std::pair<int, LayerType>>& vec1,
//     const std::vector<std::pair<int, LayerType>>& vec2) {
//   for (int i = 0; i < std::min(vec1.size(), vec2.size()); i++) {
//     if (vec1[i].second != vec2[i].second) {
//       vec1.erase(vec1.begin() + i);
//     }
//   }
//   if (vec1.size() > vec2.size()) {
//     vec1.resize(vec2.size());
//   }
// }

bool check_child(const Graph& graph, const Graph& subgraph, int i, int iter,
                 int recursionDepth) {
  int amount_connected1 = graph.getVertexValue(i + 1) - graph.getVertexValue(i);
  int amount_connected2 =
      subgraph.getVertexValue(iter + 1) - subgraph.getVertexValue(iter);
  if (amount_connected1 != amount_connected2 && amount_connected2 != 0) {
    std::cerr << "Depth " << recursionDepth << " false by outputs\n";
    return false;
  }
  if (!layer_conditions(graph.getLayerFromID(i),
                        subgraph.getLayerFromID(iter))) {
    std::cerr << "Depth " << recursionDepth << " false by layertypes\n";
    return false;
  }
  if (amount_connected2 != 0) {
    using id_name = std::pair<int, LayerType>;
    std::vector<id_name> order_a;
    std::vector<id_name> order_b;
    for (int j = 0; j < amount_connected1; j++) {
      order_a.emplace_back(id_name(
          graph.getEdgeValue(graph.getVertexValue(i) + j),
          graph.getLayerFromID(graph.getEdgeValue(graph.getVertexValue(i) + j))
              .getName()));
      order_b.emplace_back(id_name(
          subgraph.getEdgeValue(subgraph.getVertexValue(iter) + j),
          subgraph
              .getLayerFromID(
                  subgraph.getEdgeValue(subgraph.getVertexValue(iter) + j))
              .getName()));
    }
    std::sort(order_a.begin(), order_a.end(),
              [&](id_name a1, id_name a2) { return a1.second < a2.second; });
    std::sort(order_b.begin(), order_b.end(),
              [&](id_name a1, id_name a2) { return a1.second < a2.second; });
    // ^ interested in LayerType order to prevent any shuffling for childs
    // if (first_inequally) {
    //   erase_inequality_for_first(order_a, order_b);
    // }
    for (int j = 0; j < amount_connected1; j++) {
      int id1 = graph.getEdgeValue(graph.getVertexValue(i) + j);
      int id2 = subgraph.getEdgeValue(subgraph.getVertexValue(iter) + j);
      if (graph.getInputsSize(id1) != subgraph.getInputsSize(id2)) {
        std::cerr << "Depth " << recursionDepth << " false by child inputs\n";
        return false;
      }
      bool temp = check_child(graph, subgraph, order_a[j].first,
                              order_b[j].first, recursionDepth + 1);
      if (!temp) {
        std::cerr << "Depth " << recursionDepth
                  << " false by child conditions\n";
        return false;
      }
    }
  }
  std::cerr << "Depth " << recursionDepth << " true\n";
  return true;
}

}  // namespace it_lab_ai
