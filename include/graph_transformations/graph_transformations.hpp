#include <algorithm>
#include <vector>

#include "graph/graph.hpp"
#include "layers/EWLayer.hpp"
#include "layers/Layer.hpp"
#include "layers/ConvLayer.hpp"
#include "layers_fused/ConvRelu.hpp"

namespace it_lab_ai {

struct IOOrder {
  std::vector<int> in_order;
  std::vector<int> out_order;
  void fill_empty(size_t in_size, size_t out_size) {
    if (in_order.empty()) {
      in_order.resize(in_size);
      std::iota(in_order.begin(), in_order.end(), 0);
    }
    if (out_order.empty()) {
      out_order.resize(out_size);
      std::iota(out_order.begin(), out_order.end(), 0);
    }
  }
};

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
void changed_subgraphs(const Graph& graph, const Graph& subgraph_from,
                       const std::shared_ptr<Layer>& layer_to, Graph& new_graph,
                       Tensor& out,
                       const RuntimeOptions& options = RuntimeOptions());
void changed_subgraphs(const Graph& graph, const Graph& subgraph_from,
                       const Graph& subgraph_to, Graph& new_graph, Tensor& out,
                       const RuntimeOptions& options = RuntimeOptions(),
                       IOOrder order = IOOrder());
}  // namespace it_lab_ai
