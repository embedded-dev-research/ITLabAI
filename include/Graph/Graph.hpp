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
  void Work() {
    switch (type) {
      case 1:
        for (size_t i = 0; i < primer.size(); ++i) {
          primer[i] += 1;
        }
        break;
      case 2:
        for (size_t i = 0; i < primer.size(); ++i) {
          primer[i] *= 2;
        }
        break;
    }
  }
  std::vector<int> Out() { return primer; }
  /*Layer(int inputs, int neurons) : numInputs(inputs), numNeurons(neurons) {
    weights.resize(numNeurons, std::vector<double>(numInputs));
    initializeWeights();
  }
  void initializeWeights() {
    for (int i = 0; i < numNeurons; ++i) {
      for (int j = 0; j < numInputs; ++j) {
        weights[i][j] = ((double)rand() / RAND_MAX) - 0.5;
      }
    }
  }
  double activationFunction(double x) { return 1.0; }
  std::vector<double> forwardPropagation(const std::vector<double>& inputs) {
    std::vector<double> output(numNeurons, 0.0);
    for (int i = 0; i < numNeurons; ++i) {
      double neuronOutput = 0.0;
      for (int j = 0; j < numInputs; ++j) {
        neuronOutput += weights[i][j] * inputs[j];
      }
      output[i] = activationFunction(neuronOutput);
    }
    return output;
  }*/
};

class Graph {
  int BiggestSize;
  int V;
  std::vector<Layer> layers;
  std::vector<int> arrayV;
  std::vector<int> arrayE;

 public:
  Graph(int vertices) : BiggestSize(vertices) {
    if (BiggestSize < 0) {
      throw std::out_of_range("Vertices cannot be less than zero");
    }
    arrayV.push_back(0);
    V = 0;
  }
  void addEdge(int i, int j) {
    if (i == j) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < arrayV.size() - i - 1; ind++) arrayV[i + ind]++;
    arrayE.insert(arrayE.begin() + arrayV[i], j);
    arrayV[V] = arrayE.size();
  }
  void addLayer(Layer lay) {
    layers.push_back(lay);
    if (V == 0) {
      arrayV.push_back(0);
    } else {
      arrayV[V] = arrayV[V - 1];
      arrayV.push_back(arrayE.size());
    }
    V++;
  }
  bool areLayerNext(int ind1, int ind2) {
    for (int i = arrayV[ind1]; i < arrayV[ind1 + 1]; i++) {
      if (arrayE[i] == ind2) {
        return true;
      }
    }
    return false;
  }
  void checkArrays() {
    for (size_t i = 0; i < arrayV.size() - 1; ++i) {
      std::cout << arrayV[i] << " ";
    }
    std::cout << " " << arrayV[arrayV.size() - 1];
    std::cout << "\n";
    for (size_t i = 0; i < arrayE.size(); ++i) {
      std::cout << arrayE[i] << " ";
    }
    std::cout << "\n";
  }
  std::vector<int> BreadthFirstSearch(int start, int last) {
    std::queue<int> q;
    std::vector<bool> visited(V, false);
    std::vector<int> parent(V, -1);
    std::vector<int> res;
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
      int current = q.front();
      q.pop();
      if (current == last) {
        int node = current;
        while (node != -1) {
          res.push_back(node);
          node = parent[node];
        }
        /*std::cout << "iz " << start << " v " << last << " : ";
        for (int i = res.size() - 1; i >= 0; --i) {
          std::cout << res[i] << " ";
        }*/
        for(size_t i = 0; i < res.size() / 2; ++i) {
          std::swap(res[i], res[res.size() - i - 1]);
        }
        return res;
      }
      for (int ind = arrayV[current]; ind < arrayV[current + 1]; ind++) {
        int neighbor = arrayE[ind];
        if (!visited[neighbor]) {
          q.push(neighbor);
          visited[neighbor] = true;
          parent[neighbor] = current;
        }
      }
    }
  }
  std::vector<int> TraversalGraph(std::vector<int> startvec,
                                  std::vector<int> pathlayers) {
    std::vector<int> res=startvec;
    for (size_t i = 0; i < pathlayers.size(); ++i) {
      layers[pathlayers[i]].In(res);
      layers[pathlayers[i]].Work();
      res = layers[pathlayers[i]].Out();
    }
    return res;
  }
};
