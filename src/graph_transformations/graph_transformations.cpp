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
  for (int prev_id = 0; prev_id < static_cast<int>(cur_size); prev_id++) {
    int amount_connected_s =
        subgraph.getVertexValue(prev_id + 1) - subgraph.getVertexValue(prev_id);
    for (int j = 0; j < amount_connected_s; j++) {
      int next_id = subgraph.getEdgeValue(subgraph.getVertexValue(prev_id) + j);
      if (next_id < static_cast<int>(cur_size)) {
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

Graph change_subgraphs(const Graph& graph, const Graph& subgraph_from) {
  Graph new_graph = graph;
  std::vector<std::vector<int>> subs = find_subgraphs(graph, subgraph_from);
  std::vector<int> roots;
  std::vector<int> leafs;
  std::vector<int> roots_final;
  std::vector<int> leafs_final;
  std::shared_ptr<Layer> layer = std::make_shared<EWLayer>("relu");
  int amount_connected;
  for (int v = 0; v < subgraph_from.getLayersCount(); v++) {
    if (is_root(subgraph_from, v)) {
      roots.push_back(v);
    }
    if (is_leaf(subgraph_from, v)) {
      leafs.push_back(v);
    }
  }
  for (int i = 0; i < subs.size(); i++) {
    roots_final.clear();
    leafs_final.clear();
    for (int j = 0; j < roots.size(); j++) {
      // recognize transformations we can apply with roots
      amount_connected = graph.getVertexValue(subs[i][roots[j]] + 1) -
                         graph.getVertexValue(subs[i][roots[j]]);
      for (int k = 0; k < amount_connected; k++) {
        int id = graph.getEdgeValue(graph.getVertexValue(subs[i][roots[j]]) + k);
        auto it = std::find(subs[i].begin(), subs[i].end(), id);
        if (it != subs[i].end()) {
          // create copy of root
        }
      }
      // subgraph -> single node
      std::vector<int> root_inps = graph.getInLayers(subs[i][roots[j]]);

      for (int k = 0; k < root_inps.size(); k++) {
        auto it =
            std::find(roots_final.begin(), roots_final.end(), root_inps[k]);
        if (it == roots_final.end()) {
          roots_final.push_back(root_inps[k]);
        }
      }
    }
    for (int j = 0; j < leafs.size(); j++) {
      amount_connected = graph.getVertexValue(subs[i][leafs[j]] + 1) -
                         graph.getVertexValue(subs[i][leafs[j]]);
      for (int k = 0; k < amount_connected; k++) {
        int id =
            graph.getEdgeValue(graph.getVertexValue(subs[i][leafs[j]]) + k);
        auto it = std::find(leafs_final.begin(), leafs_final.end(), id);
        if (it == leafs_final.end()) {
          leafs_final.push_back(id);
        }
      }
    }
    for (int j = 0; j < subs[i].size(); j++) {
      new_graph.removeSingleLayer(subs[i][j]);
    }
    for (int j = 0; j < roots_final.size(); j++) {
      new_graph.makeConnection(new_graph.getLayerFromID(roots_final[j]), *layer);
    }
    for (int j = 0; j < leafs_final.size(); j++) {
      new_graph.makeConnection(*layer,
                               new_graph.getLayerFromID(leafs_final[j]));
    }
  }
  return new_graph;
}

}  // namespace it_lab_ai
