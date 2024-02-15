#pragma once
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

template <typename ValueType>
class Graph {
  int BiggestSize_;
  int V_;
  std::vector<Layer<ValueType>*> layers_;
  std::vector<int> arrayV_;
  std::vector<int> arrayE_;
  std::vector<ValueType> startvec_;
  std::vector<ValueType>* outvector_;
  int start_;
  int end_;

 public:
  Graph(int vertices) : BiggestSize_(vertices) {
    if (BiggestSize_ < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
    arrayV_.push_back(0);
    V_ = 0;
  }
  void setInput(Layer<ValueType>& lay, const std::vector<ValueType>& vec) {
    lay.giveID(0);
    layers_.push_back(&lay);
    arrayV_.push_back(0);
    startvec_ = vec;
    start_ = lay.checkID();
    V_++;
  }
  void makeConnection(const Layer<ValueType>& layPrev,
                      Layer<ValueType>& layNext) {
    layNext.giveID(V_);
    layers_.push_back(&layNext);
    arrayV_[V_] = arrayV_[V_ - 1];
    arrayV_.push_back(arrayE_.size());
    if (layPrev.checkID() == layNext.checkID()) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < arrayV_.size() - layPrev.checkID() - 1; ind++)
      arrayV_[layPrev.checkID() + ind]++;
    arrayE_.insert(arrayE_.begin() + arrayV_[layPrev.checkID()],
                   layNext.checkID());
    V_++;
    arrayV_[V_] = arrayE_.size();
  }
  bool areLayerNext(const Layer<ValueType>& layPrev,
                    const Layer<ValueType>& layNext) {
    for (int i = arrayV_[layPrev.checkID()]; i < arrayV_[layPrev.checkID() + 1];
         i++) {
      if (arrayE_[i] == layNext.checkID()) {
        return true;
      }
    }
    return false;
  }
  void inference() {
    std::queue<int> q;
    std::vector<bool> visited(V_, false);
    std::vector<int> parent(V_, -1);
    std::vector<int> traversal;
    q.push(start_);
    visited[start_] = true;
    while (!q.empty()) {
      int current = q.front();
      q.pop();
      if (current == end_) {
        int node = current;
        while (node != -1) {
          traversal.push_back(node);
          node = parent[node];
        }
        std::reverse(traversal.begin(), traversal.end());
        break;
      }
      for (int ind = arrayV_[current]; ind < arrayV_[current + 1]; ind++) {
        int neighbor = arrayE_[ind];
        if (!visited[neighbor]) {
          q.push(neighbor);
          visited[neighbor] = true;
          parent[neighbor] = current;
        }
      }
    }
    for (int i : traversal) {
      startvec_ = layers_[i]->run(startvec_);
    }
    outvector_->assign(startvec_.begin(), startvec_.end());
  }
  void setOutput(const Layer<ValueType>& lay, std::vector<ValueType>& vec) {
    end_ = lay.checkID();
    outvector_ = &vec;
  }
};
