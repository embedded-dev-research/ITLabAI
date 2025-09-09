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
  int V_;
  std::vector<Layer*> layers_;
  std::vector<int> arrayV_;
  std::vector<int> arrayE_;
  std::vector<Tensor> inten_;
  std::vector<Tensor> outten_;
  Tensor* outtenres_;
  int start_;
  int end_;
  std::list<BranchState> branch_list_;
  std::vector<std::vector<int>> in_edges_;
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

  void setInput(Layer& lay, Tensor& vec) {
    lay.setID(0);
    layers_.push_back(&lay);
    arrayV_.push_back(0);
    inten_ = {vec};
    start_ = lay.getID();
    V_++;
    in_edges_.resize(1);
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

    // DEBUG: Print traversal order and in/out degrees
    std::cout << "=== INFERENCE DEBUG START ===" << std::endl;
    std::cout << "Traversal order with names: ";
    for (int layer_id : traversal) {
      std::string layer_name = "unknown";
      if (layer_id >= 0 && layer_id < layers_.size()) {
        layer_name = layers_[layer_id]->getName();
      }
      std::cout << layer_id << "(" << layer_name << ") ";
    }
    std::cout << std::endl;

    std::cout << "In/Out degrees: " << std::endl;
    for (size_t i = 0; i < countinout.size(); ++i) {
      std::string layer_name = "unknown";
      if (i < layers_.size()) {
        layer_name = layers_[i]->getName();
      }
      std::cout << "Layer " << i << " (" << layer_name
                << "): " << countinout[i].first << " in, "
                << countinout[i].second << " out" << std::endl;
    }

    for (size_t i = 0; i < traversal.size(); ++i) {
      int current_layer = traversal[i];
      std::string current_layer_name = "unknown";
      if (current_layer >= 0 && current_layer < layers_.size()) {
        current_layer_name = layers_[current_layer]->getName();
      }

#ifdef ENABLE_STATISTIC_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif

      // DEBUG: Print current layer info
      std::cout << "\n--- Processing layer " << current_layer << " ("
                << current_layer_name << ") ---" << std::endl;
      std::cout << "Step " << i << "/" << traversal.size() - 1 << std::endl;

      if (i != 0) {
        std::cout << "Clearing inten_, preparing inputs..." << std::endl;
        inten_.clear();

        // DEBUG: Print input edges with layer names
        std::cout << "Input edges for layer " << current_layer << " ("
                  << current_layer_name << "): ";
        for (size_t k = 0; k < in_edges_[current_layer].size(); ++k) {
          int source_layer = in_edges_[current_layer][k];
          std::string source_name = "unknown";
          if (source_layer >= 0 && source_layer < layers_.size()) {
            source_name = layers_[source_layer]->getName();
          }
          std::cout << source_layer << "(" << source_name << ") ";
        }
        std::cout << std::endl;

        for (size_t k = 0; k < in_edges_[current_layer].size(); ++k) {
          auto target_value = in_edges_[current_layer][k];
          std::string source_name = "unknown";
          if (target_value >= 0 && target_value < layers_.size()) {
            source_name = layers_[target_value]->getName();
          }

          std::cout << "Looking for input from layer " << target_value << " ("
                    << source_name << ")" << std::endl;

          auto it = std::find_if(branch_list_.rbegin(), branch_list_.rend(),
                                 [target_value](const BranchState& s) {
                                   return s.ind_layer == target_value;
                                 });

          if (it != branch_list_.rend()) {
            std::string branch_layer_name = "unknown";
            if (it->ind_layer >= 0 && it->ind_layer < layers_.size()) {
              branch_layer_name = layers_[it->ind_layer]->getName();
            }

            std::cout << "Found branch state for layer " << target_value << " ("
                      << branch_layer_name
                      << "), distribution size: " << it->distribution.size()
                      << ", give_for_all size: " << it->give_for_all.size()
                      << std::endl;

            for (size_t f = 0; f < it->distribution.size(); ++f) {
              if (it->distribution[f].first == current_layer) {
                std::cout << "Adding tensor from distribution index " << f
                          << " to inten_" << std::endl;
                inten_.push_back(it->give_for_all[it->distribution[f].second]);
              }
            }
          } else {
            std::cout << "WARNING: No branch state found for layer "
                      << target_value << " (" << source_name << ")"
                      << std::endl;
          }

          if (it != branch_list_.rend()) {
            it->count_used_ten--;
            std::string branch_layer_name = "unknown";
            if (it->ind_layer >= 0 && it->ind_layer < layers_.size()) {
              branch_layer_name = layers_[it->ind_layer]->getName();
            }

            std::cout << "Decremented count_used_ten to " << it->count_used_ten
                      << " for layer " << target_value << " ("
                      << branch_layer_name << ")" << std::endl;

            if (it->count_used_ten < 1) {
              std::cout << "Removing branch state for layer " << target_value
                        << " (" << branch_layer_name << ")" << std::endl;
              auto rit = std::next(it).base();
              it =
                  std::reverse_iterator<decltype(rit)>(branch_list_.erase(rit));
            }
          }
        }
      }

      // DEBUG: Print input tensors before layer execution
      std::cout << "Input tensors before layer " << current_layer << " ("
                << current_layer_name << ") execution: " << inten_.size()
                << std::endl;
      for (size_t t = 0; t < inten_.size(); ++t) {
        std::cout << "  Tensor " << t << ": shape [";
        for (size_t d = 0; d < inten_[t].get_shape().dims(); ++d) {
          std::cout << inten_[t].get_shape()[d];
          if (d < inten_[t].get_shape().dims() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }

      try {
        std::cout << "Executing layer " << current_layer << " ("
                  << current_layer_name << ")..." << std::endl;
        layers_[current_layer]->run(inten_, outten_);
        std::cout << "Layer " << current_layer << " (" << current_layer_name
                  << ") execution completed successfully" << std::endl;
      } catch (const std::exception& e) {
        std::cerr << "ERROR in layer " << current_layer << " ("
                  << current_layer_name << "): " << e.what() << std::endl;
        throw;
      }

#ifdef ENABLE_STATISTIC_TENSORS
      tensors_.push_back(inten_[0]);
      tensors_.push_back(outten_[0]);
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
      weights_.push_back(layers_[i]->get_weights());
#endif

      std::cout << "Output tensors from layer " << current_layer << " ("
                << current_layer_name << "): " << outten_.size() << std::endl;
      for (size_t t = 0; t < outten_.size(); ++t) {
        std::cout << "  Output tensor " << t << ": shape [";
        for (size_t d = 0; d < outten_[t].get_shape().dims(); ++d) {
          std::cout << outten_[t].get_shape()[d];
          if (d < outten_[t].get_shape().dims() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }

      inten_ = outten_;

      if (layers_[current_layer]->postops.count > 0) {
        std::cout << "Processing " << layers_[current_layer]->postops.count
                  << " post-operations" << std::endl;
        for (unsigned int j = 0; j < layers_[current_layer]->postops.count;
             j++) {
          try {
            layers_[current_layer]->postops.layers[j]->run(inten_, outten_);
          } catch (const std::exception& e) {
            std::cerr << "ERROR in post-op " << j << " of layer "
                      << current_layer << ": " << e.what() << std::endl;
            throw;
          }
        }
        inten_ = outten_;
      }

      // Create new branch state
      BranchState new_branch;
      new_branch.give_for_all = inten_;
      new_branch.count_used_ten = countinout[current_layer].second;
      new_branch.ind_layer = current_layer;
      new_branch.split = layers_[current_layer]->getName() == kSplit;

      std::cout << "Creating branch state for layer " << current_layer
                << ": count_used_ten=" << new_branch.count_used_ten
                << ", split=" << new_branch.split << std::endl;

      if (layers_[current_layer]->getName() == kSplit) {
        std::cout << "Split layer detected" << std::endl;
        if (static_cast<int>(split_distribution_.size()) == 0) {
          std::vector<std::pair<int, int>> dis(
              countinout[current_layer].second);
          for (size_t m = 0; m < dis.size(); ++m) {
            dis[m] = {arrayE_[arrayV_[current_layer] + m], static_cast<int>(m)};
          }
          new_branch.distribution = dis;
          std::cout << "Created new distribution for split" << std::endl;
        } else {
          new_branch.distribution =
              split_distribution_[count_used_split_distribution_];
          count_used_split_distribution_++;
          std::cout << "Using pre-defined distribution "
                    << count_used_split_distribution_ - 1 << std::endl;
        }
      } else {
        std::vector<std::pair<int, int>> dis(countinout[current_layer].second);
        for (size_t m = 0; m < dis.size(); ++m) {
          dis[m] = {arrayE_[arrayV_[current_layer] + m], 0};
        }
        new_branch.distribution = dis;
      }

      // DEBUG: Print distribution
      std::cout << "Distribution: ";
      for (const auto& dist : new_branch.distribution) {
        std::cout << "(" << dist.first << "," << dist.second << ") ";
      }
      std::cout << std::endl;

      branch_list_.push_back(new_branch);

      std::cout << "Current branch list size: " << branch_list_.size()
                << std::endl;
      for (const auto& branch : branch_list_) {
        std::string branch_layer_name = "unknown";
        if (branch.ind_layer >= 0 && branch.ind_layer < layers_.size()) {
          branch_layer_name = layers_[branch.ind_layer]->getName();
        }
        std::cout << "  Layer " << branch.ind_layer << " (" << branch_layer_name
                  << ") (count: " << branch.count_used_ten
                  << ", split: " << branch.split << ")" << std::endl;
      }

#ifdef ENABLE_STATISTIC_TIME
      auto end = std::chrono::high_resolution_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      time_.push_back(static_cast<int>(elapsed.count()));
      time_layer_.push_back(layers_[i]->getName());
#endif
    }

    std::cout << "=== INFERENCE COMPLETED ===" << std::endl;
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
