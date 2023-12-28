#pragma once
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

enum LayerType {
  kInput,
  kPooling,
  kNormalization,
  kDropout,
  kElementWise,
  kConvolution,
  kFullyConnected
};

class LayerExample {
 private:
  int id_;
  std::string name_;
  LayerType type_;
  std::string version_;
  int numInputs_;
  int numNeurons_;
  std::vector<int> primer_;

 public:
  LayerExample(int id1, LayerType type1) : id_(id1), type_(type1) {}
  int checkID() const { return id_; }
  void In(const std::vector<int>& a) { primer_ = a; }
  void Work() {}
  std::vector<int> Out() { return primer_; }
};

class Graph {
  int BiggestSize_;
  int V_;
  std::vector<LayerExample> layers_;
  std::vector<int> arrayV_;
  std::vector<int> arrayE_;
  std::vector<int> outvector_;
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
  void setInput(const LayerExample& lay, const std::vector<int>& vec) {
    layers_.push_back(lay);
    arrayV_.push_back(0);
    outvector_ = vec;
    start_ = lay.checkID();
    V_++;
  }
  void makeConnection(const LayerExample& layPrev,
                      const LayerExample& layNext) {
    layers_.push_back(layNext);
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
  bool areLayerNext(const LayerExample& layPrev, const LayerExample& layNext) {
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
    end_ = V_ - 1;
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
      layers_[i].In(outvector_);
      layers_[i].Work();
      outvector_ = layers_[i].Out();
    }
  }
  std::vector<int> getOutput() { return outvector_; }
};
