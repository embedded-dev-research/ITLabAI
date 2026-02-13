#include "graph_transformations/graph_transformations.hpp"

namespace it_lab_ai {

namespace {

bool layer_conditions(const std::shared_ptr<Layer>& layer,
                      const std::shared_ptr<Layer>& layer_sub) {
  return layer->getName() == layer_sub->getName();
}

}  // namespace

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
  std::vector<int> outs = graph.getOutLayers(id_from);
  auto it = std::find(outs.begin(), outs.end(), id_to);
  return it != outs.end();
}

bool is_root(const Graph& graph, int id) {
  return graph.getInputsSize(id) == 0;
}

bool is_leaf(const Graph& graph, int id) {
  return graph.getOutputsSize(id) == 0;
}

bool run_search(const Graph& graph, const Graph& subgraph,
                std::vector<int>& assignments,
                std::vector<std::vector<int>>& results) {
  size_t cur_size = assignments.size();
  for (int prev_id = 0; prev_id < static_cast<int>(cur_size); prev_id++) {
    size_t amount_connected_s = subgraph.getOutputsSize(prev_id);
    for (size_t j = 0; j < amount_connected_s; j++) {
      int next_id = subgraph.getOutLayers(prev_id)[j];
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
            size_t amount_connected_s1 = subgraph.getOutputsSize(ids[k]);
            size_t amount_connected_1 =
                graph.getOutputsSize(assignments[ids[k]]);
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
    // special root->root case
    std::vector<int> roots;
    for (int v = 0; v < subgraph.getLayersCount(); v++) {
      if (is_root(subgraph, v)) {
        roots.push_back(assignments[v]);
      }
    }
    for (int root : roots) {
      std::vector<int> outs = graph.getOutLayers(root);
      for (int out : outs) {
        auto it = std::find(roots.begin(), roots.end(), out);
        if (it != roots.end()) {
          return false;
        }
      }
    }
    //
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
  for (auto& i : vec) {
    std::transform(i.begin(), i.end(), i.begin(),
                   [&](int elem) { return elem > id ? elem - 1 : elem; });
  }
}

bool does_intersect(const std::vector<int>& vec1,
                    const std::vector<int>& vec2) {
  // exists elem in vec1 which is found in vec2
  return std::any_of(vec1.begin(), vec1.end(), [&](int elem) {
    return std::find(vec2.begin(), vec2.end(), elem) != vec2.end();
  });
}

void changed_subgraphs(const Graph& graph, const Graph& subgraph_from,
                       Graph& new_graph, Tensor& out,
                       const RuntimeOptions& options) {
  graph.clone(new_graph, out, options);
  std::vector<std::vector<int>> subs = find_subgraphs(graph, subgraph_from);
  std::vector<std::vector<int>> subs_c = subs;
  std::vector<bool> sub_used(subs.size(), true);
  std::vector<int> roots;
  std::vector<int> leafs;
  std::vector<int> roots_inps_final;
  std::vector<int> leafs_outs_final;
  size_t amount_connected;
  size_t amount_connected_s;
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
      if (sub_used[j] && does_intersect(subs_c[j], subs_c[i])) {
        flag = true;
        break;
      }
    }
    if (flag) {
      sub_used[i] = false;
      continue;
    }
    std::shared_ptr<Layer> layer = std::make_shared<EWLayer>("relu");
    std::vector<bool> is_root_special(roots.size(), false);
    roots_inps_final.clear();
    leafs_outs_final.clear();
    for (size_t j = 0; j < roots.size(); j++) {
      std::vector<int> root_inps = new_graph.getInLayers(subs[i][roots[j]]);
      // want subgraph -> single node
      for (int root_inp : root_inps) {
        auto it = std::find(roots_inps_final.begin(), roots_inps_final.end(),
                            root_inp);
        if (it == roots_inps_final.end()) {
          roots_inps_final.push_back(root_inp);
        }
      }
      // recognize transformations we can apply with roots
      amount_connected = new_graph.getOutputsSize(subs[i][roots[j]]);
      amount_connected_s = subgraph_from.getOutputsSize(roots[j]);
      if (amount_connected == amount_connected_s) {
        continue;
      }
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][roots[j]])[k];
        auto it = std::find(subs[i].begin(), subs[i].end(), id);
        if (it == subs[i].end()) {
          is_root_special[j] = true;
        }
      }
    }
    for (int leaf : leafs) {
      amount_connected = new_graph.getOutputsSize(subs[i][leaf]);
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][leaf])[k];
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
    for (int j : roots_inps_final) {
      new_graph.makeConnection(new_graph.getLayerFromID(j), layer);
    }
    if (roots_inps_final.empty()) {
      new_graph.addSingleLayer(layer);
    }
    for (int j : leafs_outs_final) {
      new_graph.makeConnection(layer, new_graph.getLayerFromID(j));
    }
  }
}

void changed_subgraphs(const Graph& graph, const Graph& subgraph_from,
                       const Graph& subgraph_to, Graph& new_graph, Tensor& out,
                       const RuntimeOptions& options, IOOrder order) {
  graph.clone(new_graph, out, options);
  std::vector<std::vector<int>> subs = find_subgraphs(graph, subgraph_from);
  std::vector<std::vector<int>> subs_c = subs;
  std::vector<bool> sub_used(subs.size(), true);
  std::vector<int> roots;
  std::vector<int> leafs;
  std::vector<int> roots2;
  std::vector<int> leafs2;
  std::vector<std::vector<int>> roots_inps_final;
  std::vector<std::vector<int>> leafs_outs_final;
  size_t amount_connected;
  size_t amount_connected_s;
  for (int v = 0; v < subgraph_from.getLayersCount(); v++) {
    if (is_root(subgraph_from, v)) {
      roots.push_back(v);
    }
    if (is_leaf(subgraph_from, v)) {
      leafs.push_back(v);
    }
  }
  for (int v = 0; v < subgraph_to.getLayersCount(); v++) {
    if (is_root(subgraph_to, v)) {
      roots2.push_back(v);
    }
    if (is_leaf(subgraph_to, v)) {
      leafs2.push_back(v);
    }
  }
  if (roots2.size() != roots.size()) {
    throw std::invalid_argument(
        "Subgraph_to and Subgraph_from roots amounts aren't same.");
  }
  if (leafs2.size() != leafs.size()) {
    throw std::invalid_argument(
        "Subgraph_to and Subgraph_from leafs amounts aren't same.");
  }
  order.fill_empty(roots.size(), leafs.size());
  if (order.in_order.size() != roots.size()) {
    throw std::invalid_argument("Order for roots isn't complete");
  }
  if (order.out_order.size() != leafs.size()) {
    throw std::invalid_argument("Order for leafs isn't complete");
  }
  for (size_t i = 0; i < subs.size(); i++) {
    bool flag = false;
    // don't change already changed subgraph
    for (size_t j = 0; j < i; j++) {
      if (sub_used[j] && does_intersect(subs_c[j], subs_c[i])) {
        flag = true;
        break;
      }
    }
    if (flag) {
      sub_used[i] = false;
      continue;
    }
    std::vector<bool> is_root_special(roots.size(), false);
    roots_inps_final =
        std::vector<std::vector<int>>(roots.size(), std::vector<int>());
    leafs_outs_final =
        std::vector<std::vector<int>>(leafs.size(), std::vector<int>());
    for (size_t j = 0; j < roots.size(); j++) {
      roots_inps_final[j] = new_graph.getInLayers(subs[i][roots[j]]);
      // recognize transformations we can apply with roots
      amount_connected = new_graph.getOutputsSize(subs[i][roots[j]]);
      amount_connected_s = subgraph_from.getOutputsSize(roots[j]);
      if (amount_connected == amount_connected_s) {
        continue;
      }
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][roots[j]])[k];
        auto it = std::find(subs[i].begin(), subs[i].end(), id);
        if (it == subs[i].end()) {
          is_root_special[j] = true;
        }
      }
    }
    for (size_t j = 0; j < leafs.size(); j++) {
      amount_connected = new_graph.getOutputsSize(subs[i][leafs[j]]);
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][leafs[j]])[k];
        leafs_outs_final[j].push_back(id);
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
        for (auto& k : roots_inps_final) {
          std::transform(k.begin(), k.end(), k.begin(), [&](int elem) {
            return elem > subs[i][j] ? elem - 1 : elem;
          });
        }
        for (auto& k : leafs_outs_final) {
          std::transform(k.begin(), k.end(), k.begin(), [&](int elem) {
            return elem > subs[i][j] ? elem - 1 : elem;
          });
        }
      }
    }
    std::vector<int> roots2_c = roots2;
    std::vector<int> leafs2_c = leafs2;
    std::vector<std::shared_ptr<Layer>> layers;
    for (int j = 0; j < subgraph_to.getLayersCount(); j++) {
      std::shared_ptr<Layer> layer =
          layer_based_shared_copy(subgraph_to.getLayerFromID(j), options);
      layers.push_back(layer);
      new_graph.addSingleLayer(layer);
      auto it = std::find(roots2_c.begin(), roots2_c.end(), j);
      if (it != roots2_c.end()) {
        size_t index_for_root = std::distance(roots2_c.begin(), it);
        roots2[index_for_root] = layer->getID();
      }
      it = std::find(leafs2_c.begin(), leafs2_c.end(), j);
      if (it != leafs2_c.end()) {
        size_t index_for_leaf = std::distance(leafs2_c.begin(), it);
        leafs2[index_for_leaf] = layer->getID();
      }
    }
    for (int j = 0; j < subgraph_to.getLayersCount(); j++) {
      std::vector<int> cur_outs = subgraph_to.getOutLayers(j);
      for (size_t k = 0; k < cur_outs.size(); k++) {
        new_graph.makeConnection(layers[j], layers[cur_outs[k]]);
      }
    }
    for (size_t j = 0; j < roots_inps_final.size(); j++) {
      for (size_t k = 0; k < roots_inps_final[j].size(); k++) {
        new_graph.makeConnection(
            new_graph.getLayerFromID(roots_inps_final[j][k]),
            new_graph.getLayerFromID(roots2[order.in_order[j]]));
      }
    }
    for (size_t j = 0; j < leafs_outs_final.size(); j++) {
      for (size_t k = 0; k < leafs_outs_final[j].size(); k++) {
        new_graph.makeConnection(
            new_graph.getLayerFromID(leafs2[order.out_order[j]]),
            new_graph.getLayerFromID(leafs_outs_final[j][k]));
      }
    }
  }
}

}  // namespace it_lab_ai
