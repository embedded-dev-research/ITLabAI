#pragma once
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <queue>
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
    {kBatchNormalization, "Normalization"}};

struct LayerTimeStats {
  std::string layer_name;
  double total_time = 0.0;
  int call_count = 0;
  double min_time = std::numeric_limits<double>::max();
  double max_time = 0.0;
};

struct BranchState {
  int ind_layer;
  std::vector<Tensor> give_for_all;
  int count_used_ten;
  bool split;
  std::vector<std::pair<int, int>> distribution;
};

std::shared_ptr<Layer> layer_based_shared_copy(
    const std::shared_ptr<Layer>& layer, const RuntimeOptions& options);

class Graph {
  std::map<std::string, LayerTimeStats> layer_stats_;
  int BiggestSize_;
  int V_;  // amount of ids
  std::vector<std::shared_ptr<Layer>> layers_;
  std::vector<int> arrayV_;  // vertices (id -> vertex number)
  std::vector<int> arrayE_;  // edges (vertex number -> id)
  std::vector<Tensor> inten_;
  std::vector<Tensor> outten_;
  Tensor* outtenres_;
  int start_;
  int end_;
  std::unordered_map<int, BranchState> branch_map_;
  std::vector<std::vector<int>> in_edges_;  // next -> prev
  std::vector<std::vector<std::pair<int, int>>> split_distribution_;
  int count_used_split_distribution_;
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
  Graph() {
    arrayV_.push_back(0);
    V_ = 0;
    in_edges_.clear();
  }

  Graph(int vertices, std::vector<std::vector<std::pair<int, int>>> split)
      : BiggestSize_(vertices), split_distribution_(std::move(split)) {
    if (BiggestSize_ < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
    arrayV_.push_back(0);
    V_ = 0;
    in_edges_.clear();
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
  }

  [[nodiscard]] int getVertexValue(size_t layerID) const {
    if (layerID >= arrayV_.size()) {
      throw std::invalid_argument("ArrayV does not contain this ID.");
    }
    return arrayV_[layerID];
  }

  [[nodiscard]] int getEdgeValue(size_t pos) const {
    if (pos >= arrayE_.size()) {
      throw std::invalid_argument("ArrayE does not contain this.");
    }
    return arrayE_[pos];
  }

  [[nodiscard]] size_t getInputsSize(size_t layerID) const {
    if (layerID >= in_edges_.size()) {
      throw std::invalid_argument("Input edges array do not contain this ID.");
    }
    return in_edges_[layerID].size();
  }

  [[nodiscard]] std::vector<int> getInLayers(size_t layerID) const {
    if (layerID >= in_edges_.size()) {
      throw std::invalid_argument("Input edges array do not contain this ID.");
    }
    return in_edges_[layerID];
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
  }
  void removeSingleLayer(int id) {
    if (id >= V_ || id < 0) {
      throw std::out_of_range("Layer ID out of range");
    }
    // remove inputs
    for (int i = 0; i < V_; i++) {
      if (arrayV_[i] != arrayV_[i + 1]) {
        auto array_e_it = std::find(arrayE_.begin() + arrayV_[i],
                                    arrayE_.begin() + arrayV_[i + 1], id);
        if (array_e_it != arrayE_.begin() + arrayV_[i + 1]) {
          removeConnection(i, id);
        }
      }
    }
    // remove outputs
    int amount_connected = arrayV_[id + 1] - arrayV_[id];
    for (int i = 0; i < amount_connected; i++) {
      removeConnection(id, arrayE_[arrayV_[id] + i]);
    }
    // remove vertex
    in_edges_.erase(in_edges_.begin() + id);
    arrayV_.erase(arrayV_.begin() + id);
    for (int& i : arrayE_) {
      if (i > id) {
        i -= 1;
      }
    }
    for (std::vector<int>& i : in_edges_) {
      for (int& j : i) {
        if (j > id) {
          j--;
        }
      }
    }
    for (size_t i = id + 1; i < layers_.size(); i++) {
      layers_[i]->setID(layers_[i]->getID() - 1);
    }
    layers_[id]->setID(-1);
    layers_.erase(layers_.begin() + id);
    V_--;
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
    std::vector<std::pair<int, int>> countinout = getInOutDegrees();
    std::vector<int> traversal = getTraversalOrder();
    count_used_split_distribution_ = 0;

    for (size_t i = 0; i < traversal.size(); ++i) {
      int current_layer = traversal[i];
#ifdef ENABLE_STATISTIC_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif
      if (i != 0) {
        inten_.clear();

        for (size_t k = 0; k < in_edges_[current_layer].size(); ++k) {
          auto target_value = in_edges_[current_layer][k];
          auto it = branch_map_.find(target_value);

          if (it != branch_map_.end()) {
            for (size_t f = 0; f < it->second.distribution.size(); ++f) {
              if (it->second.distribution[f].first == current_layer) {
                bool last_use = (it->second.count_used_ten == 1);
                auto& src =
                    it->second.give_for_all[it->second.distribution[f].second];
                if (last_use) {
                  inten_.push_back(std::move(src));
                } else {
                  inten_.push_back(src);
                }
              }
            }

            it->second.count_used_ten--;
            if (it->second.count_used_ten < 1) {
              branch_map_.erase(it);
            }
          }
        }
      }
      if (outten_.empty()) {
        outten_.resize(1);
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

      BranchState new_branch;
      new_branch.give_for_all = std::move(inten_);
      new_branch.count_used_ten = countinout[current_layer].second;
      new_branch.ind_layer = current_layer;
      new_branch.split = layers_[current_layer]->getName() == kSplit;

      if (layers_[current_layer]->getName() == kSplit) {
        if (static_cast<int>(split_distribution_.size()) == 0) {
          std::vector<std::pair<int, int>> dis(
              countinout[current_layer].second);
          for (size_t m = 0; m < dis.size(); ++m) {
            dis[m] = {arrayE_[arrayV_[current_layer] + m], static_cast<int>(m)};
          }
          new_branch.distribution = dis;
        } else {
          new_branch.distribution =
              split_distribution_[count_used_split_distribution_];
          count_used_split_distribution_++;
        }
      } else {
        std::vector<std::pair<int, int>> dis(countinout[current_layer].second);
        for (size_t m = 0; m < dis.size(); ++m) {
          dis[m] = {arrayE_[arrayV_[current_layer] + m], 0};
        }
        new_branch.distribution = dis;
      }
      branch_map_[current_layer] = std::move(new_branch);
      if (outtenres_ && current_layer == end_ &&
          !branch_map_[current_layer].give_for_all.empty() &&
          countinout[current_layer].second == 0) {
        *outtenres_ = std::move(branch_map_[current_layer].give_for_all[0]);
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
    if (outten_.empty()) {
      std::vector<int> vec1 = {1, 7, 1, 0};
      Tensor start = make_tensor(vec1);
      outten_.push_back(start);
    }
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
    std::vector<int> in_degree(V_, 0);

    for (int i = 0; i < V_; ++i) {
      for (int j = arrayV_[i]; j < arrayV_[i + 1]; ++j) {
        int target_vertex = arrayE_[j];
        if (target_vertex >= 0 && target_vertex < V_) {
          in_degree[target_vertex]++;
        }
      }
    }

    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < V_; ++i) {
      int out_degree = arrayV_[i + 1] - arrayV_[i];
      result.emplace_back(in_degree[i], out_degree);
    }

    return result;
  }

  void printLayerStats() {
    std::cout << "\n========== LAYER PERFORMANCE STATISTICS ==========\n";
    std::cout << std::left << std::setw(20) << "Layer Type" << std::right
              << std::setw(15) << "Total (ms)" << std::setw(12) << "Calls"
              << std::setw(15) << "Avg (ms)" << std::setw(15) << "Min (ms)"
              << std::setw(15) << "Max (ms)" << std::endl;

    for (const auto& [name, stats] : layer_stats_) {
      double avg = stats.total_time / stats.call_count;
      std::cout << std::left << std::setw(20) << name << std::right
                << std::fixed << std::setprecision(3) << std::setw(15)
                << stats.total_time << std::setw(12) << stats.call_count
                << std::setw(15) << avg << std::setw(15) << stats.min_time
                << std::setw(15) << stats.max_time << std::endl;
    }
  }

  [[nodiscard]] std::vector<int> getTraversalOrder() const {
    auto in_out_degrees = getInOutDegrees();
    std::vector<int> in_degree(V_);
    for (int i = 0; i < V_; ++i) {
      in_degree[i] = in_out_degrees[i].first;
    }

    std::vector<int> traversal;
    std::vector<bool> visited(V_, false);

    std::function<void(int)> dfs = [&](int u) {
      if (visited[u]) return;
      visited[u] = true;
      traversal.push_back(u);

      std::vector<int> children;
      for (int j = arrayV_[u]; j < arrayV_[u + 1]; ++j) {
        int v = arrayE_[j];
        children.push_back(v);
      }

      std::sort(children.begin(), children.end());

      for (int child : children) {
        in_degree[child]--;
        if (in_degree[child] == 0 && !visited[child]) {
          dfs(child);
        }
      }
    };

    for (int i = 0; i < V_; ++i) {
      if (in_degree[i] == 0 && !visited[i]) {
        dfs(i);
      }
    }

    return traversal;
  }
};
}  // namespace it_lab_ai
