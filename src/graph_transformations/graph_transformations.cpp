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

void change_ids(std::vector<std::vector<int>>& vec, int id) {
  for (size_t i = 0; i < vec.size(); i++) {
    std::transform(vec[i].begin(), vec[i].end(), vec[i].begin(),
                   [&](int elem) { return elem > id ? elem - 1 : elem; });
  }
}

bool does_intersect(const std::vector<int>& vec1,
                    const std::vector<int>& vec2) {
  for (size_t i = 0; i < vec1.size(); i++) {
    auto it = std::find(vec2.begin(), vec2.end(), vec1[i]);
    if (it != vec2.end()) {
      return true;
    }
  }
  return false;
}

Graph changed_subgraphs(const Graph& graph, const Graph& subgraph_from) {
  Graph new_graph = graph;
  std::vector<std::vector<int>> subs = find_subgraphs(graph, subgraph_from);
  std::vector<std::vector<int>> subs_c = subs;
  std::vector<int> roots;
  std::vector<int> leafs;
  std::vector<int> roots_inps_final;
  std::vector<int> leafs_outs_final;
  int amount_connected;
  int amount_connected_s;
  for (int v = 0; v < subgraph_from.getLayersCount(); v++) {
    if (is_root(subgraph_from, v)) {
      roots.push_back(v);
    }
    if (is_leaf(subgraph_from, v)) {
      leafs.push_back(v);
    }
  }
  for (size_t i = 0; i < subs.size(); i++) {
    bool flag = false;
    // don't change already changed subgraph
    for (size_t j = 0; j < i; j++) {
      if (does_intersect(subs_c[j], subs_c[i])) {
        flag = true;
        break;
      }
    }
    if (flag) {
      continue;
    }
    std::shared_ptr<Layer> layer = std::make_shared<EWLayer>("relu");
    std::vector<bool> is_root_special(roots.size(), false);
    roots_inps_final.clear();
    leafs_outs_final.clear();
    for (size_t j = 0; j < roots.size(); j++) {
      std::vector<int> root_inps = new_graph.getInLayers(subs[i][roots[j]]);
      // recognize transformations we can apply with roots
      amount_connected = new_graph.getVertexValue(subs[i][roots[j]] + 1) -
                         new_graph.getVertexValue(subs[i][roots[j]]);
      amount_connected_s = subgraph_from.getVertexValue(roots[j] + 1) -
                           subgraph_from.getVertexValue(roots[j]);
      if (amount_connected == amount_connected_s) {
        continue;
      }
      for (int k = 0; k < amount_connected; k++) {
        int id = new_graph.getEdgeValue(
            new_graph.getVertexValue(subs[i][roots[j]]) + k);
        auto it = std::find(subs[i].begin(), subs[i].end(), id);
        if (it == subs[i].end()) {
          is_root_special[j] = true;
        }
      }

      // want subgraph -> single node
      for (size_t k = 0; k < root_inps.size(); k++) {
        auto it = std::find(roots_inps_final.begin(), roots_inps_final.end(),
                            root_inps[k]);
        if (it == roots_inps_final.end()) {
          roots_inps_final.push_back(root_inps[k]);
        }
      }
    }
    for (size_t j = 0; j < leafs.size(); j++) {
      amount_connected = new_graph.getVertexValue(subs[i][leafs[j]] + 1) -
                         new_graph.getVertexValue(subs[i][leafs[j]]);
      for (int k = 0; k < amount_connected; k++) {
        int id = new_graph.getEdgeValue(
            new_graph.getVertexValue(subs[i][leafs[j]]) + k);
        auto it =
            std::find(leafs_outs_final.begin(), leafs_outs_final.end(), id);
        if (it == leafs_outs_final.end()) {
          leafs_outs_final.push_back(id);
        }
      }
    }
    for (size_t j = 0; j < subs[i].size(); j++) {
      auto it = std::find(roots.begin(), roots.end(), j);
      size_t index_for_root = std::distance(roots.begin(), it);
      // remove all nodes that isn't special roots
      if (it == roots.end() ||
          (it != roots.end() && !is_root_special[index_for_root])) {
        new_graph.removeSingleLayer(subs[i][j]);
        change_ids(subs, subs[i][j]);
        std::transform(roots_inps_final.begin(), roots_inps_final.end(),
                       roots_inps_final.begin(), [&](int elem) {
                         return elem > subs[i][j] ? elem - 1 : elem;
                       });
        std::transform(leafs_outs_final.begin(), leafs_outs_final.end(),
                       leafs_outs_final.begin(), [&](int elem) {
                         return elem > subs[i][j] ? elem - 1 : elem;
                       });
      }
    }
    for (size_t j = 0; j < roots_inps_final.size(); j++) {
      new_graph.makeConnection(new_graph.getLayerFromID(roots_inps_final[j]),
                               *layer);
    }
    if (roots_inps_final.size() == 0) {
      new_graph.addSingleLayer(*layer);
    }
    for (size_t j = 0; j < leafs_outs_final.size(); j++) {
      new_graph.makeConnection(*layer,
                               new_graph.getLayerFromID(leafs_outs_final[j]));
    }
  }
  return new_graph;
}

}  // namespace it_lab_ai
