#pragma once
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/Layer.hpp"

namespace itlab_2023 {

class Graph {
  int BiggestSize_;
  int V_;
  std::vector<Layer*> layers_;
  std::vector<int> arrayV_;
  std::vector<int> arrayE_;
  Tensor inten_;
  Tensor* outten_;
  int start_;
  int end_;
  std::vector<Tensor> tensors_;

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
    inten_ = vec;
    start_ = lay.getID();
    V_++;
  }
  void makeConnection(const Layer& layPrev, Layer& layNext) {
    layNext.setID(V_);
    layers_.push_back(&layNext);
    arrayV_[V_] = arrayV_[V_ - 1];
    arrayV_.push_back(static_cast<int>(arrayE_.size()));
    if (layPrev.getID() == layNext.getID()) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < static_cast<int>(arrayV_.size()) -
                                static_cast<int>(layPrev.getID()) - 1;
         ind++)
      arrayV_[layPrev.getID() + ind]++;
    arrayE_.insert(arrayE_.begin() + arrayV_[layPrev.getID()], layNext.getID());
    V_++;
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
      layers_[i]->run(inten_, *outten_);
      tensors_.push_back(*outten_);
      inten_ = *outten_;
    }
  }
  void setOutput(const Layer& lay, Tensor& vec) {
    end_ = lay.getID();
    outten_ = &vec;
  }
  std::vector<Tensor> getTensors() { return tensors_; }
};
}  // namespace itlab_2023
