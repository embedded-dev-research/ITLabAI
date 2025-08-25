#pragma once
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <stack>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "layers/Layer.hpp"

namespace it_lab_ai {

struct BranchState {
  std::vector<Tensor> give_for_all;
  std::vector<Tensor> buf;
  uint8_t count_used_ten;
  bool split;
  uint8_t ind_layer;
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
  std::stack<BranchState> branch_stack;
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
  }
  void setInput(Layer& lay, Tensor& vec) {
    lay.setID(0);
    layers_.push_back(&lay);
    arrayV_.push_back(0);
    inten_ = {vec};
    start_ = lay.getID();
    V_++;
  }
  void makeConnection(const Layer& layPrev, Layer& layNext) {
    bool layerExists = false;
    for (const auto* layer : layers_) {
      if (layer == &layNext) {
        layerExists = true;
        break;
      }
    }

    if (!layerExists) {
      layNext.setID(V_);
      layers_.push_back(&layNext);
      arrayV_.push_back(static_cast<int>(arrayE_.size()));
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
    // for (size_t i = 0; i < countinout.size(); ++i) {
    //   std::cout << "Vertex " << i << ": in=" << countinout[i].first
    //             << ", out=" << countinout[i].second << std::endl;
    // }
    // for (size_t i = 0; i < traversal.size(); ++i) {
    //   std::cout << traversal[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << std::endl;

    for (size_t i = 0; i < traversal.size(); ++i) {
#ifdef ENABLE_STATISTIC_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif
      if (countinout[traversal[i]].first > 1) {
        inten_ = branch_stack.top().buf;
        if (static_cast<int>(inten_.size()) < countinout[traversal[i]].first) {
          BranchState& top_branch = branch_stack.top();
          bool check = 0;
          if (arrayE_[arrayV_[top_branch.ind_layer]] == traversal[i]) {
            std::vector<Tensor> r = {top_branch.give_for_all[0]};
            for (size_t k = 0; k < inten_.size(); k++) {
              r.push_back(inten_[k]);
            }
            inten_ = r;
          } else {
            for (int i1 = 1; i1 < arrayV_[top_branch.ind_layer + 1] -
                                      arrayV_[top_branch.ind_layer];
                 ++i1)
              if (arrayE_[arrayV_[top_branch.ind_layer] + i1] == traversal[i])
                check = 1;
          }
          if (check) {
            if (!top_branch.split)
              inten_.push_back(top_branch.give_for_all[0]);
            else {
              inten_.push_back(
                  top_branch.give_for_all[top_branch.give_for_all.size() - 1]);
            }
          }
        }
        branch_stack.pop();
        while (static_cast<int>(inten_.size()) <
               countinout[traversal[i]].first) {
          std::vector<Tensor> r = branch_stack.top().buf;
          for (size_t k = 0; k < inten_.size(); k++) {
            r.push_back(inten_[k]);
          }
          inten_ = r;
          if (static_cast<int>(inten_.size()) <
              countinout[traversal[i]].first) {
            BranchState& top_branch = branch_stack.top();
            bool check = 0;
            if (arrayE_[arrayV_[top_branch.ind_layer]] == traversal[i]) {
              std::vector<Tensor> r1 = {top_branch.give_for_all[0]};
              for (size_t k = 0; k < inten_.size(); k++) {
                r1.push_back(inten_[k]);
              }
              inten_ = r1;
            } else {
              for (int i1 = 1; i1 < arrayV_[top_branch.ind_layer + 1] -
                                        arrayV_[top_branch.ind_layer];
                   ++i1)
                if (arrayE_[arrayV_[top_branch.ind_layer] + i1] == traversal[i])
                  check = 1;
            }
            if (check) {
              if (!top_branch.split)
                inten_.push_back(top_branch.give_for_all[0]);
              else {
                inten_.push_back(
                    top_branch
                        .give_for_all[top_branch.give_for_all.size() - 1]);
              }
            }
          }
          branch_stack.pop();
        }
      } else {
        if (countinout[traversal[i]].first != 0) {
          if (countinout[arrayE_[arrayV_[traversal[i - 1]]]].first > 1) {
            BranchState& top_branch = branch_stack.top();
            if (!top_branch.split) {
              inten_ = top_branch.give_for_all;
            } else {
              inten_ = {top_branch.give_for_all[top_branch.count_used_ten]};
              top_branch.count_used_ten++;
            }
          } else if (layers_[traversal[i - 1]]->getName() == kSplit) {
            BranchState& top_branch = branch_stack.top();
            if (top_branch.count_used_ten == 0) {
              inten_ = {top_branch.give_for_all[top_branch.count_used_ten]};
              top_branch.count_used_ten++;
            }
          }
        }
      }
      //   std::cout << "inten_" << std::endl;
      // for (size_t m = 0; m < inten_.size(); ++m) {
      //   std::cout << inten_[m] << " ";
      // }
      // std::cout << std::endl;
      layers_[traversal[i]]->run(inten_, outten_);
      //   std::cout << "outten_" << std::endl;
      // for (size_t m = 0; m < outten_.size(); ++m) {
      //   std::cout << outten_[m] << " ";
      // }
      // std::cout << std::endl;

#ifdef ENABLE_STATISTIC_TENSORS
      tensors_.push_back(inten_[0]);
      tensors_.push_back(outten_[0]);
#endif
#ifdef ENABLE_STATISTIC_WEIGHTS
      weights_.push_back(layers_[i]->get_weights());
#endif

      inten_ = outten_;
      if (outten_.size() > 1) outten_.resize(1);
      if (layers_[traversal[i]]->postops.count > 0) {
        for (unsigned int j = 0; j < layers_[traversal[i]]->postops.count;
             j++) {
          layers_[traversal[i]]->postops.layers[j]->run(inten_, outten_);
        }
        inten_ = outten_;
      }
      if (countinout[traversal[i]].second == 1 &&
          countinout[arrayE_[arrayV_[traversal[i]]]].first > 1) {
        BranchState& top_branch = branch_stack.top();
        top_branch.buf.push_back(inten_[0]);
      }
      if (countinout[traversal[i]].second > 1) {
        BranchState new_branch;
        new_branch.give_for_all = inten_;
        new_branch.count_used_ten = 0;
        new_branch.ind_layer = static_cast<uint8_t>(traversal[i]);
        if (layers_[traversal[i]]->getName() == kSplit) {
          new_branch.split = 1;
        } else {
          new_branch.split = 0;
        }
        branch_stack.push(new_branch);
      }

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
    std::vector<int> inDegree(V_, 0);

    for (int i = 0; i < V_; ++i) {
      for (int j = arrayV_[i]; j < arrayV_[i + 1]; ++j) {
        int targetVertex = arrayE_[j];
        if (targetVertex >= 0 && targetVertex < V_) {
          inDegree[targetVertex]++;
        }
      }
    }

    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < V_; ++i) {
      int outDegree = arrayV_[i + 1] - arrayV_[i];
      result.emplace_back(inDegree[i], outDegree);
    }

    return result;
  }
  std::vector<int> getTraversalOrder() const {
    auto inOutDegrees = getInOutDegrees();
    std::vector<int> inDegree(V_);
    for (int i = 0; i < V_; ++i) {
      inDegree[i] = inOutDegrees[i].first;
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
        inDegree[child]--;
        if (inDegree[child] == 0 && !visited[child]) {
          dfs(child);
        }
      }
    };

    for (int i = 0; i < V_; ++i) {
      if (inDegree[i] == 0 && !visited[i]) {
        dfs(i);
      }
    }

    return traversal;
  }
};
}  // namespace it_lab_ai
