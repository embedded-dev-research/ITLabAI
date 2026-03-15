#pragma once
#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "layers/Layer.hpp"
#include "runtime_options.hpp"

namespace it_lab_ai {
static std::unordered_map<LayerType, std::string> label_map = {
    {kInput, "Input"},
    {kPooling, "Pooling"},
    {kElementWise, "Element-wise"},
    {kConvolution, "Convolution"},
    {kFullyConnected, "Dense"},
    {kFlatten, "Flatten"},
    {kConcat, "Concat"},
    {kDropout, "Dropout"},
    {kSplit, "Split"},
    {kBinaryOp, "BinaryOp"},
    {kTranspose, "Transpose"},
    {kMatmul, "MatMul"},
    {kReshape, "Reshape"},
    {kSoftmax, "Softmax"},
    {kReduce, "Reduce"},
    {kBatchNormalization, "Normalization"},
    {kConvRelu, "ConvRelu"}};

struct LayerTimeStats {
  std::string layer_name;
  double total_time = 0.0;
  int call_count = 0;
  double min_time = std::numeric_limits<double>::max();
  double max_time = 0.0;
};

struct BranchState {
  std::vector<Tensor> tensors;
  int remaining_uses = 0;
  bool active = false;
};

struct InputBinding {
  int source_layer = -1;
  std::vector<int> output_slots;
};

std::shared_ptr<Layer> layer_based_shared_copy(
    const std::shared_ptr<Layer>& layer, const RuntimeOptions& options);

class Graph {
  using Route = std::pair<int, int>;

  std::map<std::string, LayerTimeStats> layer_stats_;
  int BiggestSize_ = 0;
  int V_ = 0;  // amount of ids
  std::vector<std::shared_ptr<Layer>> layers_;
  std::vector<int> arrayV_;  // vertices (id -> vertex number)
  std::vector<int> arrayE_;  // edges (vertex number -> id)
  std::vector<Tensor> inten_;
  std::vector<Tensor> outten_;
  Tensor* outtenres_ = nullptr;
  int start_ = -1;
  int end_ = -1;
  std::vector<std::vector<int>> in_edges_;  // next -> prev
  std::vector<std::vector<std::pair<int, int>>> split_distribution_;
  mutable bool execution_plan_dirty_ = true;
  mutable std::vector<std::pair<int, int>> in_out_degrees_;
  mutable std::vector<int> traversal_order_;
  mutable std::vector<std::vector<Route>> output_routes_;
  mutable std::vector<std::vector<InputBinding>> input_bindings_;
  mutable std::vector<size_t> expected_input_count_;
  mutable std::vector<BranchState> branch_states_;
#ifdef ENABLE_STATISTIC_TENSORS
  std::vector<Tensor> tensors_;
#endif
#ifdef ENABLE_STATISTIC_TIME
  std::vector<int> time_;
  std::vector<LayerType> time_layer_;
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
  std::vector<Tensor> weights_;
#endif

 public:
  Graph() { arrayV_.push_back(0); }

  Graph(int vertices, std::vector<std::vector<std::pair<int, int>>> split)
      : BiggestSize_(vertices), split_distribution_(std::move(split)) {
    if (BiggestSize_ < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
    arrayV_.push_back(0);
  }

  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;
  Graph(Graph&&) noexcept = default;
  Graph& operator=(Graph&&) noexcept = default;
  ~Graph() = default;

  void clone(Graph& result, Tensor& out,
             const RuntimeOptions& options = RuntimeOptions()) const;

  void setSplitDistribution(
      std::vector<std::vector<std::pair<int, int>>> split_dist) {
    split_distribution_ = std::move(split_dist);
    markExecutionPlanDirty();
  }

  [[nodiscard]] size_t getInputsSize(size_t layerID) const {
    if (layerID >= in_edges_.size()) {
      throw std::invalid_argument(
          "Input edges array does not contain this ID.");
    }
    return in_edges_[layerID].size();
  }

  [[nodiscard]] std::vector<int> getInLayers(size_t layerID) const {
    if (layerID >= in_edges_.size()) {
      throw std::invalid_argument(
          "Input edges array does not contain this ID.");
    }
    return in_edges_[layerID];
  }

  [[nodiscard]] size_t getOutputsSize(size_t layerID) const {
    if (layerID >= layers_.size()) {
      throw std::invalid_argument("Layers array does not contain this ID.");
    }
    return arrayV_[layerID + 1] - arrayV_[layerID];
  }

  [[nodiscard]] std::vector<int> getOutLayers(size_t layerID) const {
    if (layerID >= layers_.size()) {
      throw std::invalid_argument(
          "Output edges array does not contain this ID.");
    }
    return std::vector<int>(arrayE_.begin() + arrayV_[layerID],
                            arrayE_.begin() + arrayV_[layerID + 1]);
  }

  [[nodiscard]] int getLayersCount() const { return V_; }

  [[nodiscard]] std::shared_ptr<Layer> getLayerFromID(size_t layerID) const {
    if (layerID >= layers_.size()) {
      throw std::invalid_argument("Layers do not contain this ID.");
    }
    return layers_[layerID];
  }

  void setInput(const std::shared_ptr<Layer>& layer, Tensor& vec) {
    if (!layer) {
      throw std::invalid_argument("Layer cannot be null");
    }

    int previous_start = start_;
    int id = layer->getID();
    bool layer_exists = (id >= 0 && id < V_ && layers_[id] == layer);

    if (!layer_exists) {
      layer->setID(V_);
      layers_.emplace_back(layer);
      arrayV_.push_back(static_cast<int>(arrayE_.size()));

      if (V_ >= static_cast<int>(in_edges_.size())) {
        in_edges_.resize(V_ + 1);
      }

      V_++;
    }

    inten_ = {vec};
    start_ = layer->getID();
    if (!layer_exists || start_ != previous_start) {
      markExecutionPlanDirty();
    }
  }

  void addSingleLayer(const std::shared_ptr<Layer>& layer) {
    if (!layer) return;

    int id = layer->getID();
    bool layer_exists = (id >= 0 && id < V_ && layers_[id] == layer);

    if (!layer_exists) {
      layer->setID(V_);
      layers_.push_back(layer);
      arrayV_.push_back(static_cast<int>(arrayE_.size()));

      if (V_ >= static_cast<int>(in_edges_.size())) {
        in_edges_.resize(V_ + 1);
      }

      V_++;
      markExecutionPlanDirty();
    }
  }

  void makeConnection(const std::shared_ptr<Layer>& layPrev,
                      const std::shared_ptr<Layer>& layNext) {
    if (!layPrev || !layNext) {
      throw std::invalid_argument("Layers cannot be null");
    }

    addSingleLayer(layPrev);
    addSingleLayer(layNext);

    if (layPrev->getID() == layNext->getID()) {
      throw std::out_of_range("i=j cant add edge");
    }

    for (int i = arrayV_[layPrev->getID()]; i < arrayV_[layPrev->getID() + 1];
         ++i) {
      if (arrayE_[i] == layNext->getID()) {
        return;
      }
    }

    for (int i = layPrev->getID() + 1; i < V_; ++i) {
      arrayV_[i]++;
    }
    arrayE_.insert(arrayE_.begin() + arrayV_[layPrev->getID()],
                   layNext->getID());
    arrayV_[V_] = static_cast<int>(arrayE_.size());

    if (layNext->getID() >= static_cast<int>(in_edges_.size())) {
      in_edges_.resize(layNext->getID() + 1);
    }

    in_edges_[layNext->getID()].push_back(layPrev->getID());
    markExecutionPlanDirty();
  }

  void removeConnection(int idPrev, int idNext) {
    if (idPrev >= V_ || idNext >= V_ || idPrev < 0 || idNext < 0) {
      throw std::out_of_range("Layer ID out of range");
    }
    auto it =
        std::find(in_edges_[idNext].begin(), in_edges_[idNext].end(), idPrev);
    if (it == in_edges_[idNext].end()) {
      throw std::invalid_argument(
          (std::string("No such edge ") + std::to_string(idPrev)) + " " +
          std::to_string(idNext));
    }
    in_edges_[idNext].erase(it);
    auto array_e_it = std::find(arrayE_.begin() + arrayV_[idPrev],
                                arrayE_.begin() + arrayV_[idPrev + 1], idNext);
    if (array_e_it == arrayE_.begin() + arrayV_[idPrev + 1]) {
      throw std::invalid_argument(
          (std::string("No such edge ") + std::to_string(idPrev)) + " " +
          std::to_string(idNext));
    }
    arrayE_.erase(array_e_it);
    for (size_t i = static_cast<size_t>(idPrev) + 1; i < arrayV_.size(); ++i) {
      arrayV_[i]--;
    }
    markExecutionPlanDirty();
  }

  void removeSingleLayer(int id) {
    if (id >= V_ || id < 0) {
      throw std::out_of_range("Layer ID out of range");
    }

    for (int i = 0; i < V_; i++) {
      if (arrayV_[i] != arrayV_[i + 1]) {
        auto array_e_it = std::find(arrayE_.begin() + arrayV_[i],
                                    arrayE_.begin() + arrayV_[i + 1], id);
        if (array_e_it != arrayE_.begin() + arrayV_[i + 1]) {
          removeConnection(i, id);
        }
      }
    }

    int amount_connected = arrayV_[id + 1] - arrayV_[id];
    std::vector<int> array_e_copy = arrayE_;
    for (int i = 0; i < amount_connected; i++) {
      removeConnection(id, array_e_copy[arrayV_[id] + i]);
    }

    in_edges_.erase(in_edges_.begin() + id);
    arrayV_.erase(arrayV_.begin() + id);
    for (int& edge : arrayE_) {
      if (edge > id) {
        edge -= 1;
      }
    }
    for (std::vector<int>& edges : in_edges_) {
      for (int& edge : edges) {
        if (edge > id) {
          edge--;
        }
      }
    }
    for (size_t i = id + 1; i < layers_.size(); i++) {
      layers_[i]->setID(layers_[i]->getID() - 1);
    }
    layers_[id]->setID(-1);
    layers_.erase(layers_.begin() + id);
    V_--;
    markExecutionPlanDirty();
  }

  bool areLayerNext(const std::shared_ptr<Layer>& layPrev,
                    const std::shared_ptr<Layer>& layNext) {
    if (!layPrev || !layNext) return false;

    if (layPrev->getID() >= V_ || layPrev->getID() < 0) {
      throw std::invalid_argument("No such layer in graph");
    }

    for (int i = arrayV_[layPrev->getID()]; i < arrayV_[layPrev->getID() + 1];
         i++) {
      if (arrayE_[i] == layNext->getID()) {
        return true;
      }
    }
    return false;
  }

  void inference(const RuntimeOptions& options) {
    ensureExecutionPlan();

    if (outten_.empty()) {
      outten_.resize(1);
    }

    for (int layer_id : traversal_order_) {
      auto& branch = branch_states_[layer_id];
      branch.tensors.clear();
      branch.remaining_uses = 0;
      branch.active = false;
    }

    for (size_t order_index = 0; order_index < traversal_order_.size();
         ++order_index) {
      const int current_layer = traversal_order_[order_index];
#ifdef ENABLE_STATISTIC_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif
      if (order_index != 0) {
        inten_.clear();
        inten_.reserve(expected_input_count_[current_layer]);

        for (const auto& binding : input_bindings_[current_layer]) {
          auto& source_state = branch_states_[binding.source_layer];
          if (!source_state.active) {
            continue;
          }

          const bool last_use = (source_state.remaining_uses == 1);
          for (int output_slot : binding.output_slots) {
            auto& src = source_state.tensors[static_cast<size_t>(output_slot)];
            if (last_use) {
              inten_.push_back(std::move(src));
            } else {
              inten_.push_back(src);
            }
          }

          source_state.remaining_uses--;
          if (source_state.remaining_uses < 1) {
            source_state.tensors.clear();
            source_state.active = false;
          }
        }
      }

      layers_[current_layer]->run(inten_, outten_, options);

#ifdef ENABLE_STATISTIC_TENSORS
      tensors_.push_back(inten_[0]);
      tensors_.push_back(outten_[0]);
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
      weights_.push_back(layers_[current_layer]->get_weights());
#endif

      inten_.swap(outten_);

      if (layers_[current_layer]->postops.count > 0) {
        for (unsigned int j = 0; j < layers_[current_layer]->postops.count;
             j++) {
          layers_[current_layer]->postops.layers[j]->run(inten_, outten_,
                                                         options);
        }
        inten_.swap(outten_);
      }

      auto& current_branch = branch_states_[current_layer];
      current_branch.tensors = std::move(inten_);
      current_branch.remaining_uses = in_out_degrees_[current_layer].second;
      current_branch.active = current_branch.remaining_uses > 0;

      if (current_branch.remaining_uses == 0) {
        if (outtenres_ && current_layer == end_ &&
            !current_branch.tensors.empty()) {
          *outtenres_ = std::move(current_branch.tensors[0]);
        }
        current_branch.tensors.clear();
      }

#ifdef ENABLE_STATISTIC_TIME
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      int elapsed_ms = static_cast<int>(elapsed.count());
      time_.push_back(elapsed_ms);

      LayerType layer_type = layers_[current_layer]->getName();
      time_layer_.push_back(layer_type);

      auto it = label_map.find(layer_type);
      std::string layer_name_str =
          (it != label_map.end()) ? it->second : "Unknown";

      auto& stats = layer_stats_[layer_name_str];
      stats.total_time += elapsed_ms;
      stats.call_count++;

      if (stats.call_count == 1) {
        stats.min_time = elapsed_ms;
        stats.max_time = elapsed_ms;
      } else {
        if (elapsed_ms < stats.min_time) stats.min_time = elapsed_ms;
        if (elapsed_ms > stats.max_time) stats.max_time = elapsed_ms;
      }
#endif
    }
  }

  void setOutput(const std::shared_ptr<Layer>& layer, Tensor& vec) {
    if (!layer) {
      throw std::invalid_argument("Layer cannot be null");
    }
    end_ = layer->getID();
    outtenres_ = &vec;
  }

#ifdef ENABLE_STATISTIC_TENSORS
  std::vector<Tensor> getTensors() { return tensors_; }
#endif
#ifdef ENABLE_STATISTIC_TIME
  std::vector<std::string> getTimeInfo() {
    std::vector<std::string> res;
    for (size_t i = 0; i < time_.size(); i++) {
      auto it = label_map.find(time_layer_[i]);
      std::string layer_name = (it != label_map.end()) ? it->second : "Unknown";
      res.push_back(layer_name + ':' + std::to_string(time_[i]));
    }
    return res;
  }
  std::vector<int> getTime() { return time_; }
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
  std::vector<Tensor> getWEIGHTS() { return weights_; }
#endif

  [[nodiscard]] std::vector<std::pair<int, int>> getInOutDegrees() const {
    ensureExecutionPlan();
    return in_out_degrees_;
  }

  void printLayerStats() {
    std::cout << "\n========== LAYER PERFORMANCE STATISTICS ==========\n";
    std::cout << std::left << std::setw(20) << "Layer Type" << std::right
              << std::setw(15) << "Total (ms)" << std::setw(12) << "Calls"
              << std::setw(15) << "Avg (ms)" << std::setw(15) << "Min (ms)"
              << std::setw(15) << "Max (ms)" << '\n';

    for (const auto& [name, stats] : layer_stats_) {
      double avg = stats.total_time / stats.call_count;
      std::cout << std::left << std::setw(20) << name << std::right
                << std::fixed << std::setprecision(3) << std::setw(15)
                << stats.total_time << std::setw(12) << stats.call_count
                << std::setw(15) << avg << std::setw(15) << stats.min_time
                << std::setw(15) << stats.max_time << '\n';
    }
  }

  [[nodiscard]] std::vector<int> getTraversalOrder() const {
    ensureExecutionPlan();
    return traversal_order_;
  }

 private:
  void markExecutionPlanDirty() { execution_plan_dirty_ = true; }

  void ensureExecutionPlan() const {
    if (execution_plan_dirty_) {
      rebuildExecutionPlan();
    }
  }

  [[nodiscard]] std::vector<Route> buildLayerRoutes(
      int layer_id, size_t& split_distribution_index) const {
    const int out_degree = arrayV_[layer_id + 1] - arrayV_[layer_id];
    std::vector<Route> routes(static_cast<size_t>(out_degree));

    if (layers_[layer_id]->getName() == kSplit) {
      if (split_distribution_.empty()) {
        for (int edge_index = 0; edge_index < out_degree; ++edge_index) {
          routes[static_cast<size_t>(edge_index)] = {
              arrayE_[arrayV_[layer_id] + edge_index], edge_index};
        }
      } else {
        if (split_distribution_index >= split_distribution_.size()) {
          throw std::out_of_range(
              "Split distribution does not match split layer count");
        }
        routes = split_distribution_[split_distribution_index++];
      }
    } else {
      for (int edge_index = 0; edge_index < out_degree; ++edge_index) {
        routes[static_cast<size_t>(edge_index)] = {
            arrayE_[arrayV_[layer_id] + edge_index], 0};
      }
    }

    return routes;
  }

  void rebuildExecutionPlan() const {
    in_out_degrees_.assign(static_cast<size_t>(V_), {0, 0});
    std::vector<int> in_degree(static_cast<size_t>(V_), 0);

    for (int layer_id = 0; layer_id < V_; ++layer_id) {
      const int out_degree = arrayV_[layer_id + 1] - arrayV_[layer_id];
      in_out_degrees_[static_cast<size_t>(layer_id)].second = out_degree;

      for (int edge_index = arrayV_[layer_id];
           edge_index < arrayV_[layer_id + 1]; ++edge_index) {
        const int target_vertex = arrayE_[edge_index];
        if (target_vertex >= 0 && target_vertex < V_) {
          in_degree[static_cast<size_t>(target_vertex)]++;
        }
      }
    }

    for (int layer_id = 0; layer_id < V_; ++layer_id) {
      in_out_degrees_[static_cast<size_t>(layer_id)].first =
          in_degree[static_cast<size_t>(layer_id)];
    }

    traversal_order_.clear();
    traversal_order_.reserve(static_cast<size_t>(V_));
    std::vector<bool> visited(static_cast<size_t>(V_), false);
    std::vector<int> traversal_in_degree = in_degree;

    std::function<void(int)> dfs = [&](int vertex) {
      if (visited[static_cast<size_t>(vertex)]) return;

      visited[static_cast<size_t>(vertex)] = true;
      traversal_order_.push_back(vertex);

      std::vector<int> children;
      children.reserve(
          static_cast<size_t>(arrayV_[vertex + 1] - arrayV_[vertex]));
      for (int edge_index = arrayV_[vertex]; edge_index < arrayV_[vertex + 1];
           ++edge_index) {
        children.push_back(arrayE_[edge_index]);
      }

      std::sort(children.begin(), children.end());

      for (int child : children) {
        traversal_in_degree[static_cast<size_t>(child)]--;
        if (traversal_in_degree[static_cast<size_t>(child)] == 0 &&
            !visited[static_cast<size_t>(child)]) {
          dfs(child);
        }
      }
    };

    for (int layer_id = 0; layer_id < V_; ++layer_id) {
      if (traversal_in_degree[static_cast<size_t>(layer_id)] == 0 &&
          !visited[static_cast<size_t>(layer_id)]) {
        dfs(layer_id);
      }
    }

    output_routes_.assign(static_cast<size_t>(V_), {});
    size_t split_distribution_index = 0;
    for (int layer_id : traversal_order_) {
      output_routes_[static_cast<size_t>(layer_id)] =
          buildLayerRoutes(layer_id, split_distribution_index);
    }

    input_bindings_.assign(static_cast<size_t>(V_), {});
    expected_input_count_.assign(static_cast<size_t>(V_), 0);
    branch_states_.assign(static_cast<size_t>(V_), {});

    for (int layer_id = 0; layer_id < V_; ++layer_id) {
      auto& layer_bindings = input_bindings_[static_cast<size_t>(layer_id)];
      layer_bindings.reserve(in_edges_[static_cast<size_t>(layer_id)].size());

      size_t input_count = 0;
      for (int source_layer : in_edges_[static_cast<size_t>(layer_id)]) {
        InputBinding binding;
        binding.source_layer = source_layer;

        for (const auto& route :
             output_routes_[static_cast<size_t>(source_layer)]) {
          if (route.first == layer_id) {
            binding.output_slots.push_back(route.second);
          }
        }

        input_count += binding.output_slots.size();
        layer_bindings.push_back(std::move(binding));
      }

      expected_input_count_[static_cast<size_t>(layer_id)] = input_count;
    }

    execution_plan_dirty_ = false;
  }
};
}  // namespace it_lab_ai
