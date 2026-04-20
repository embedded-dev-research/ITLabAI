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
  bool refined = false;
  for (int i = 0; i < subgraph.getLayersCount(); i++) {
    if (subgraph.getOutLayers(i).size() > 1 || subgraph.getInLayers(i).size() > 1) {
      refined = true;
      break;
    }
  }
  if (refined) {
    std::vector<std::vector<int>> possible_assignments;
    fill_possible_assignments(graph, subgraph, possible_assignments);
    bool res = update_refinement(graph, subgraph, possible_assignments);
    if (res) {
      run_search_refined(graph, subgraph, assignments, results,
                         possible_assignments);
    }
  } else {
    run_search(graph, subgraph, assignments, results);
  }
  return results;
}

void fill_possible_assignments(
    const Graph& graph, const Graph& subgraph,
    std::vector<std::vector<int>>& possible_assignments) {
  possible_assignments.assign(subgraph.getLayersCount(), std::vector<int>());
  for (size_t i = 0; i < possible_assignments.size(); i++) {
    for (size_t j = 0; j < graph.getLayersCount(); j++) {
      if (graph.getOutputsSize(j) < subgraph.getOutputsSize(i) || graph.getInputsSize(j) < subgraph.getInputsSize(i)) {
        continue;
      }
      possible_assignments[i].push_back(static_cast<int>(j));
    }
  }
}

bool update_refinement(const Graph& graph, const Graph& subgraph,
                       std::vector<std::vector<int>>& possible_assignments) {
  bool has_changed = true;
  while (has_changed) {
    has_changed = false;
    for (size_t i = 0; i < possible_assignments.size(); i++) {
      for (int j = 0; j < static_cast<int>(possible_assignments[i].size());
           j++) {
        bool is_erased = false;
        for (size_t k = 0; k < subgraph.getOutputsSize(i); k++) {
          bool match = false;
          for (int x : possible_assignments[subgraph.getOutLayers(i)[k]]) {
            // candidate for i #j should be connected with at least one of
            // candidates of i's output nodes
            //std::cerr << i << " " << j << " " << k << " "
             //         << has_edge(graph, possible_assignments[i][j], x) << std::endl;
            if (has_edge(graph, possible_assignments[i][j], x)) {
              match = true;
              break;
            }
          }
          if (!match) {
            possible_assignments[i].erase(possible_assignments[i].begin() + j);
            has_changed = true;
            is_erased = true;
            j--;
            break;
          }
        }
        if (is_erased) {
          continue;
        }
        for (size_t k = 0; k < subgraph.getInputsSize(i); k++) {
          bool match = false;
          for (int x : possible_assignments[subgraph.getInLayers(i)[k]]) {
            // candidate for i #j should be connected with at least one of
            // candidates of i's input nodes
            // std::cerr << i << " " << j << " " << k << " "
            //         << has_edge(graph, possible_assignments[i][j], x) <<
            //         std::endl;
            if (has_edge(graph, x, possible_assignments[i][j])) {
              match = true;
              break;
            }
          }
          if (!match) {
            possible_assignments[i].erase(possible_assignments[i].begin() + j);
            has_changed = true;
            j--;
            break;
          }
        }
      }
    }
    for (size_t i = 0; i < possible_assignments.size(); i++) {
      if (possible_assignments[i].empty()) {
        return false;
      }
    }
  }
  return true;
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

bool run_search_refined(const Graph& graph, const Graph& subgraph,
                std::vector<int>& assignments,
                std::vector<std::vector<int>>& results,
                std::vector<std::vector<int>>& possible_assignments) {
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
  for (int i = 0;
       i < static_cast<int>(possible_assignments[cur_size].size());
       i++) {
    int id = possible_assignments[cur_size][i];
    auto it = std::find(assignments.begin(), assignments.end(), id);
    if (it == assignments.end()) {
      assignments.push_back(id);
      //auto possible_assignments_copy = possible_assignments;
      //possible_assignments_copy[cur_size] = {id};
      //if (run_search_refined(graph, subgraph, assignments, results,
      //                       possible_assignments_copy)) {
      //  results.emplace_back(assignments);
      //}
      if (run_search_refined(graph, subgraph, assignments, results,
                             possible_assignments)) {
        results.emplace_back(assignments);
      }
      assignments.pop_back();
    }
    //possible_assignments[cur_size].erase(
    //    possible_assignments[cur_size].begin() + i);
    //i--;
    //bool ret = update_refinement(graph, subgraph, possible_assignments);
    //if (!ret) {
    //  break;
    //}
  }
  return false;
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
                       const std::shared_ptr<Layer>& layer_to, Graph& new_graph,
                       Tensor& out, const RuntimeOptions& options) {
  graph.clone(new_graph, out, options);
  std::vector<std::vector<int>> subs = find_subgraphs(graph, subgraph_from);
  std::vector<std::vector<int>> subs_c = subs;
  std::vector<bool> sub_used(subs.size(), true);
  std::vector<int> roots;
  std::vector<int> leaves;
  std::vector<int> roots_inps_final;
  std::vector<int> leaves_outs_final;
  size_t amount_connected;
  size_t amount_connected_s;
  for (int v = 0; v < subgraph_from.getLayersCount(); v++) {
    if (is_root(subgraph_from, v)) {
      roots.push_back(v);
    }
    if (is_leaf(subgraph_from, v)) {
      leaves.push_back(v);
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
    std::shared_ptr<Layer> layer;
    if (layer_to->getName() == kConvRelu &&
        graph.getLayerFromID(subs_c[i][0])->getName() == kConvolution) {
      layer = std::static_pointer_cast<Layer>(std::make_shared<ConvReluLayer>(
          std::dynamic_pointer_cast<ConvolutionalLayer>(
              graph.getLayerFromID(subs_c[i][0]))));  // convrelu case
    } else if (layer_to->getName() == kDenseNetPath && subs_c[i].size() == 5 &&
               graph.getLayerFromID(subs_c[i][0])->getName() ==
                   kBatchNormalization &&
               graph.getLayerFromID(subs_c[i][2])->getName() == kConvolution &&
               graph.getLayerFromID(subs_c[i][4])->getName() == kConvolution) {
      layer = std::static_pointer_cast<Layer>(std::make_shared<DenseNetPath>(
          std::dynamic_pointer_cast<BatchNormalizationLayer>(
              graph.getLayerFromID(subs_c[i][0])),
          std::dynamic_pointer_cast<ConvolutionalLayer>(
              graph.getLayerFromID(subs_c[i][2])),
          std::dynamic_pointer_cast<ConvolutionalLayer>(
              graph.getLayerFromID(subs_c[i][4]))));  // densenetpath case
    } else {
      layer = layer_based_shared_copy(layer_to, options);
    }
    std::vector<bool> is_root_special(roots.size(), false);
    roots_inps_final.clear();
    leaves_outs_final.clear();
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
    for (int leaf : leaves) {
      amount_connected = new_graph.getOutputsSize(subs[i][leaf]);
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][leaf])[k];
        auto it =
            std::find(leaves_outs_final.begin(), leaves_outs_final.end(), id);
        if (it == leaves_outs_final.end()) {
          leaves_outs_final.push_back(id);
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
        std::transform(leaves_outs_final.begin(), leaves_outs_final.end(),
                       leaves_outs_final.begin(), [&](int elem) {
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
    for (int j : leaves_outs_final) {
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
  std::vector<int> roots_from;
  std::vector<int> leaves_from;
  std::vector<int> roots_to;
  std::vector<int> leaves_to;
  std::vector<std::vector<int>> roots_inps_final;
  std::vector<std::vector<int>> leaves_outs_final;
  size_t amount_connected;
  size_t amount_connected_s;
  for (int v = 0; v < subgraph_from.getLayersCount(); v++) {
    if (is_root(subgraph_from, v)) {
      roots_from.push_back(v);
    }
    if (is_leaf(subgraph_from, v)) {
      leaves_from.push_back(v);
    }
  }
  for (int v = 0; v < subgraph_to.getLayersCount(); v++) {
    if (is_root(subgraph_to, v)) {
      roots_to.push_back(v);
    }
    if (is_leaf(subgraph_to, v)) {
      leaves_to.push_back(v);
    }
  }
  if (roots_to.size() != roots_from.size()) {
    throw std::invalid_argument(
        "Subgraph_to and Subgraph_from roots amounts aren't same.");
  }
  if (leaves_to.size() != leaves_from.size()) {
    throw std::invalid_argument(
        "Subgraph_to and Subgraph_from leaves amounts aren't same.");
  }
  order.fill_empty(roots_from.size(), leaves_from.size());
  if (order.in_order.size() != roots_from.size()) {
    throw std::invalid_argument("Order for roots isn't complete");
  }
  if (order.out_order.size() != leaves_from.size()) {
    throw std::invalid_argument("Order for leaves isn't complete");
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
    std::vector<bool> is_root_special(roots_from.size(), false);
    roots_inps_final =
        std::vector<std::vector<int>>(roots_from.size(), std::vector<int>());
    leaves_outs_final =
        std::vector<std::vector<int>>(leaves_from.size(), std::vector<int>());
    for (size_t j = 0; j < roots_from.size(); j++) {
      roots_inps_final[j] = new_graph.getInLayers(subs[i][roots_from[j]]);
      // recognize transformations we can apply with roots
      amount_connected = new_graph.getOutputsSize(subs[i][roots_from[j]]);
      amount_connected_s = subgraph_from.getOutputsSize(roots_from[j]);
      if (amount_connected == amount_connected_s) {
        continue;
      }
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][roots_from[j]])[k];
        auto it = std::find(subs[i].begin(), subs[i].end(), id);
        if (it == subs[i].end()) {
          is_root_special[j] = true;
        }
      }
    }
    for (size_t j = 0; j < leaves_from.size(); j++) {
      amount_connected = new_graph.getOutputsSize(subs[i][leaves_from[j]]);
      for (size_t k = 0; k < amount_connected; k++) {
        int id = new_graph.getOutLayers(subs[i][leaves_from[j]])[k];
        leaves_outs_final[j].push_back(id);
      }
    }
    for (size_t j = 0; j < subs[i].size(); j++) {
      auto it = std::find(roots_from.begin(), roots_from.end(), j);
      size_t index_for_root = std::distance(roots_from.begin(), it);
      // remove all nodes that isn't special roots
      if (it == roots_from.end() ||
          (it != roots_from.end() && !is_root_special[index_for_root])) {
        new_graph.removeSingleLayer(subs[i][j]);
        change_ids(subs, subs[i][j]);
        for (auto& k : roots_inps_final) {
          std::transform(k.begin(), k.end(), k.begin(), [&](int elem) {
            return elem > subs[i][j] ? elem - 1 : elem;
          });
        }
        for (auto& k : leaves_outs_final) {
          std::transform(k.begin(), k.end(), k.begin(), [&](int elem) {
            return elem > subs[i][j] ? elem - 1 : elem;
          });
        }
      }
    }
    std::vector<int> roots_to_c = roots_to;
    std::vector<int> leaves_to_c = leaves_to;
    std::vector<std::shared_ptr<Layer>> layers;
    for (int j = 0; j < subgraph_to.getLayersCount(); j++) {
      std::shared_ptr<Layer> layer =
          layer_based_shared_copy(subgraph_to.getLayerFromID(j), options);
      layers.push_back(layer);
      new_graph.addSingleLayer(layer);
      auto it = std::find(roots_to_c.begin(), roots_to_c.end(), j);
      if (it != roots_to_c.end()) {
        size_t index_for_root = std::distance(roots_to_c.begin(), it);
        roots_to[index_for_root] = layer->getID();
      }
      it = std::find(leaves_to_c.begin(), leaves_to_c.end(), j);
      if (it != leaves_to_c.end()) {
        size_t index_for_leaf = std::distance(leaves_to_c.begin(), it);
        leaves_to[index_for_leaf] = layer->getID();
      }
    }
    for (int j = 0; j < subgraph_to.getLayersCount(); j++) {
      std::vector<int> cur_outs = subgraph_to.getOutLayers(j);
      for (int cur_out : cur_outs) {
        new_graph.makeConnection(layers[j], layers[cur_out]);
      }
    }
    for (size_t j = 0; j < roots_inps_final.size(); j++) {
      for (size_t k = 0; k < roots_inps_final[j].size(); k++) {
        new_graph.makeConnection(
            new_graph.getLayerFromID(roots_inps_final[j][k]),
            new_graph.getLayerFromID(roots_to[order.in_order[j]]));
      }
    }
    for (size_t j = 0; j < leaves_outs_final.size(); j++) {
      for (size_t k = 0; k < leaves_outs_final[j].size(); k++) {
        new_graph.makeConnection(
            new_graph.getLayerFromID(leaves_to[order.out_order[j]]),
            new_graph.getLayerFromID(leaves_outs_final[j][k]));
      }
    }
  }
}

}  // namespace it_lab_ai
