#pragma once
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

class Layer {
 private:
  int id;
  std::string name;
  int type;
  std::string version;
  int numInputs;
  int numNeurons;
  std::vector<int> primer;

 public:
  Layer() : id(0), type(1) {}
  Layer(int id1, int type1) : id(id1), type(type1) {}
  int checkID() { return id; }
  void In(std::vector<int> a) { primer = a; }
  void Work() {}
  std::vector<int> Out() { return primer; }
};

class Graph {
  int BiggestSize;
  int V;
  std::vector<Layer> layers;
  std::vector<int> arrayV;
  std::vector<int> arrayE;
  int start;
  int end;

 public:
  Graph(int vertices) : BiggestSize(vertices) {
    if (BiggestSize < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
    arrayV.push_back(0);
    V = 0;
  }
  void input(Layer lay, int end1, std::vector<int> vec) {
    layers.push_back(lay);
    arrayV.push_back(0);
    start = lay.checkID();
    end = end1;
    V++;
  }
  void conection(Layer layPrev, Layer layNext) {
    layers.push_back(layNext);
    arrayV[V] = arrayV[V - 1];
    arrayV.push_back(arrayE.size());
    if (layPrev.checkID() == layNext.checkID()) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < arrayV.size() - layPrev.checkID() - 1; ind++)
      arrayV[layPrev.checkID() + ind]++;
    arrayE.insert(arrayE.begin() + arrayV[layPrev.checkID()],
                  layNext.checkID());
    V++;
    arrayV[V] = arrayE.size();
  }
  bool areLayerNext(Layer layPrev, Layer layNext) {
    for (int i = arrayV[layPrev.checkID()]; i < arrayV[layPrev.checkID() + 1];
         i++) {
      if (arrayE[i] == layNext.checkID()) {
        return true;
      }
    }
    return false;
  }
};
