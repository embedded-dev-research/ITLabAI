#pragma once
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

class LayerExample {
 private:
  int id_;
  std::string name_;
  int type_;
  std::string version_;
  int numInputs_;
  int numNeurons_;
  std::vector<int> primer_;

 public:
<<<<<<< HEAD
  Layer(int id1, int type1) : id_(id1), type_(type1) {}
=======
  LayerExample(int id1, int type1) : id_(id1), type_(type1) {}
>>>>>>> 4aa654bd13a3b0c729db81455bfe478f26ba557a
  int checkID() const { return id_; }
  void In(const std::vector<int>& a) { primer_ = a; }
  std::vector<int> Out() { return primer_; }
};

class Graph {
  int BiggestSize_;
  int V_;
<<<<<<< HEAD
  std::vector<Layer> layers_;
=======
  std::vector<LayerExample> layers_;
>>>>>>> 4aa654bd13a3b0c729db81455bfe478f26ba557a
  std::vector<int> arrayV_;
  std::vector<int> arrayE_;
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
<<<<<<< HEAD
  void setInput(const Layer& lay, const std::vector<int>& vec) {
=======
  void setInput(const LayerExample& lay, const std::vector<int>& vec) {
>>>>>>> 4aa654bd13a3b0c729db81455bfe478f26ba557a
    layers_.push_back(lay);
    arrayV_.push_back(0);
    start_ = lay.checkID();
    V_++;
  }
<<<<<<< HEAD
  void makeConnection(const Layer& layPrev, const Layer& layNext) {
=======
  void makeConnection(const LayerExample& layPrev,
                      const LayerExample& layNext) {
>>>>>>> 4aa654bd13a3b0c729db81455bfe478f26ba557a
    layers_.push_back(layNext);
    arrayV_[V_] = arrayV_[V_ - 1];
    arrayV_.push_back(arrayE_.size());
    if (layPrev.checkID() == layNext.checkID()) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < arrayV_.size() - layPrev.checkID() - 1; ind++)
      arrayV_[layPrev.checkID() + ind]++;
    arrayE_.insert(arrayE_.begin() + arrayV_[layPrev.checkID()],
<<<<<<< HEAD
                  layNext.checkID());
    V_++;
    arrayV_[V_] = arrayE_.size();
  }
  bool areLayerNext(const Layer& layPrev, const Layer& layNext) {
=======
                   layNext.checkID());
    V_++;
    arrayV_[V_] = arrayE_.size();
  }
  bool areLayerNext(const LayerExample& layPrev, const LayerExample& layNext) {
>>>>>>> 4aa654bd13a3b0c729db81455bfe478f26ba557a
    for (int i = arrayV_[layPrev.checkID()]; i < arrayV_[layPrev.checkID() + 1];
         i++) {
      if (arrayE_[i] == layNext.checkID()) {
        return true;
      }
    }
    return false;
  }
};
