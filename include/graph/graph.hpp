#pragma once
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

enum LayerType {
  kInput,
  kPooling,
  kNormalization,
  kDropout,
  kElementWise,
  kConvolution,
  kFullyConnected,
  kOutput
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
  LayerExample(LayerType type1) : type_(type1) {}
  LayerType getType() { return type_; }
  int getNumInputs() const { return numInputs_; }
  int getNumNeurons() const { return numNeurons_; }
  int checkID() const { return id_; }
  void giveID(int id1) { id_ = id1; }
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
  std::vector<int> startvec_;
  std::vector<int>* outvector_;
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
  void setInput(LayerExample& lay, const std::vector<int>& vec) {
    lay.giveID(0);
    layers_.push_back(lay);
    arrayV_.push_back(0);
    startvec_ = vec;
    start_ = lay.checkID();
    V_++;
  }
  void makeConnection(const LayerExample& layPrev, LayerExample& layNext) {
    layNext.giveID(V_);
    layers_.push_back(layNext);
    arrayV_[V_] = arrayV_[V_ - 1];
    arrayV_.push_back(static_cast<int>(arrayE_.size()));
    if (layPrev.checkID() == layNext.checkID()) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < static_cast<int>(arrayV_.size()) -
                                static_cast<int>(layPrev.checkID()) - 1;
         ind++)
      arrayV_[layPrev.checkID() + ind]++;
    arrayE_.insert(arrayE_.begin() + arrayV_[layPrev.checkID()],
                   layNext.checkID());
    V_++;
    arrayV_[V_] = static_cast<int>(arrayE_.size());
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
      layers_[i].In(startvec_);
      layers_[i].Work();
      startvec_ = layers_[i].Out();
    }
    outvector_->assign(startvec_.begin(), startvec_.end());
  }
  void setOutput(const LayerExample& lay, std::vector<int>& vec) {
    end_ = lay.checkID();
    outvector_ = &vec;
  }
};
