#pragma once
#include <algorithm>
#include <chrono>
#include <list>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

struct BranchState {
  int ind_layer;
  std::vector<Tensor> give_for_all;
  int count_used_ten;
  bool split;
  std::vector<std::pair<int, int>> distribution;
};

class Graph {
  int BiggestSize_;
  int V_;                       // amount of ids
  std::vector<Layer*> layers_;  // layers vector with some ids
  std::vector<int> arrayV_;     // vertices (id -> vertex number)
  std::vector<int> arrayE_;     // edges (vertex number -> id)
  std::vector<Tensor> inten_;
  std::vector<Tensor> outten_;
  Tensor* outtenres_;
  int start_;
  int end_;
  std::list<BranchState> branch_list_;
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
  Graph(int vertices) : BiggestSize_(vertices) {
    if (BiggestSize_ < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
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

  int getVertexValue(size_t layerID) const {
    if (layerID >= arrayV_.size()) {
      throw std::invalid_argument("ArrayV does not contain this ID.");
    }
    return arrayV_[layerID];
  }

  int getEdgeValue(size_t pos) const {
    if (pos >= arrayE_.size()) {
      throw std::invalid_argument("ArrayE does not contain this.");
    }
    return arrayE_[pos];
  }

  size_t getInputsSize(size_t layerID) const {
    if (layerID >= in_edges_.size()) {
      throw std::invalid_argument("Input edges array do not contain this ID.");
    }
    return in_edges_[layerID].size();
  }

  int getLayersCount() const { return V_; }
  const Layer& getLayerFromID(size_t layerID) const {
    if (layerID >= layers_.size()) {
      throw std::invalid_argument("Layers do not contain this ID.");
    }
    return *layers_[layerID];
  }

  void setInput(Layer& lay, Tensor& vec) {
    lay.setID(0);
    layers_.push_back(&lay);
    arrayV_.push_back(0);
    inten_ = {vec};
    start_ = lay.getID();
    V_++;
    in_edges_.resize(1);
  }

  void addSingleLayer(Layer& lay) {
    bool layer_exists = false;
    for (const auto* layer : layers_) {
      if (layer == &lay) {
        layer_exists = true;
        break;
      }
    }

    if (!layer_exists) {
      lay.setID(V_);
      layers_.push_back(&lay);
      arrayV_.push_back(static_cast<int>(arrayE_.size()));

      if (V_ >= static_cast<int>(in_edges_.size())) {
        in_edges_.resize(V_ + 1);
      }

      V_++;
    }
  }

  void makeConnection(const Layer& layPrev, Layer& layNext) {
    bool layer_exists = false;
    for (const auto* layer : layers_) {
      if (layer == &layNext) {
        layer_exists = true;
        break;
      }
    }

    if (!layer_exists) {
      layNext.setID(V_);
      layers_.push_back(&layNext);
      arrayV_.push_back(static_cast<int>(arrayE_.size()));

      if (V_ >= static_cast<int>(in_edges_.size())) {
        in_edges_.resize(V_ + 1);
      }

      V_++;
    }

    if (layPrev.getID() == layNext.getID()) {
      throw std::out_of_range("i=j cant add edge");
    }

    for (int i = layPrev.getID() + 1; i < V_; ++i) {
      arrayV_[i]++;
    }
    arrayE_.insert(arrayE_.begin() + arrayV_[layPrev.getID()], layNext.getID());
    arrayV_[V_] = static_cast<int>(arrayE_.size());

    if (layNext.getID() >= static_cast<int>(in_edges_.size())) {
      in_edges_.resize(layNext.getID() + 1);
    }

    in_edges_[layNext.getID()].push_back(layPrev.getID());
  }
  bool areLayerNext(const Layer& layPrev, const Layer& layNext) {
    for (int i = arrayV_[layPrev.getID()]; i < arrayV_[layPrev.getID() + 1];
         i++) {
      if (arrayE_[i] == layNext.getID()) {
        return true;
      }
    }
    return false;
  }
  void inference() {
    std::vector<std::pair<int, int>> countinout = getInOutDegrees();
    std::vector<int> traversal = getTraversalOrder();
    count_used_split_distribution_ = 0;

    for (size_t i = 0; i < traversal.size(); ++i) {
#ifdef ENABLE_STATISTIC_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif
      if (i != 0) {
        inten_.clear();
        for (size_t k = 0; k < in_edges_[traversal[i]].size(); ++k) {
          auto target_value = in_edges_[traversal[i]][k];

          auto it = std::find_if(branch_list_.rbegin(), branch_list_.rend(),
                                 [target_value](const BranchState& s) {
                                   return s.ind_layer == target_value;
                                 });
          if (it != branch_list_.rend()) {
            for (size_t f = 0; f < it->distribution.size(); ++f) {
              if (it->distribution[f].first == traversal[i]) {
                inten_.push_back(it->give_for_all[it->distribution[f].second]);
              }
            }
          }
          it->count_used_ten--;
          if (it->count_used_ten < 1) {
            auto rit = std::next(it).base();
            it = std::reverse_iterator<decltype(rit)>(branch_list_.erase(rit));
          }
        }
      }
      layers_[traversal[i]]->run(inten_, outten_);

#ifdef ENABLE_STATISTIC_TENSORS
      tensors_.push_back(inten_[0]);
      tensors_.push_back(outten_[0]);
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
      weights_.push_back(layers_[i]->get_weights());
#endif

      inten_ = outten_;
      if (layers_[traversal[i]]->postops.count > 0) {
        for (unsigned int j = 0; j < layers_[traversal[i]]->postops.count;
             j++) {
          layers_[traversal[i]]->postops.layers[j]->run(inten_, outten_);
        }
        inten_ = outten_;
      }

      BranchState new_branch;
      new_branch.give_for_all = inten_;
      new_branch.count_used_ten = countinout[traversal[i]].second;
      new_branch.ind_layer = traversal[i];
      new_branch.split = layers_[traversal[i]]->getName() == kSplit;
      if (layers_[traversal[i]]->getName() == kSplit) {
        if (static_cast<int>(split_distribution_.size()) == 0) {
          std::vector<std::pair<int, int>> dis(countinout[traversal[i]].second);
          for (size_t m = 0; m < dis.size(); ++m) {
            dis[m] = {arrayE_[arrayV_[traversal[i]] + m], static_cast<int>(m)};
          }
          new_branch.distribution = dis;
        } else {
          new_branch.distribution =
              split_distribution_[count_used_split_distribution_];
          count_used_split_distribution_++;
        }
      } else {
        std::vector<std::pair<int, int>> dis(countinout[traversal[i]].second);
        for (size_t m = 0; m < dis.size(); ++m) {
          dis[m] = {arrayE_[arrayV_[traversal[i]] + m], 0};
        }
        new_branch.distribution = dis;
      }
      branch_list_.push_back(new_branch);

#ifdef ENABLE_STATISTIC_TIME
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      time_.push_back(static_cast<int>(elapsed.count()));
      time_layer_.push_back(layers_[i]->getName());
#endif
    }
    *outtenres_ = outten_[0];
  }
  void setOutput(const Layer& lay, Tensor& vec) {
    end_ = lay.getID();
    outtenres_ = &vec;
    std::vector<int> vec1 = {1, 7, 1, 0};
    Tensor start = make_tensor(vec1);
    outten_.push_back(start);
  }
#ifdef ENABLE_STATISTIC_TENSORS
  std::vector<Tensor> getTensors() { return tensors_; }
#endif
#ifdef ENABLE_STATISTIC_TIME
  std::vector<std::string> getTimeInfo() {
    std::vector<std::string> res;
    std::vector<std::string> labels = {
        "Input",       "Pooling", "Normalization", "Dropout", "Element-wise",
        "Convolution", "Dense",   "Flatten",       "Output"};
    for (size_t i = 0; i < time_.size(); i++) {
      res.push_back(labels[static_cast<size_t>(time_layer_[i])] + ':' +
                    std::to_string(time_[i]));
    }
    return res;
  }
  std::vector<int> getTime() { return time_; }
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
  std::vector<Tensor> getWEIGHTS() { return weights_; }
#endif
  std::vector<std::pair<int, int>> getInOutDegrees() const {
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
  std::vector<int> getTraversalOrder() const {
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
