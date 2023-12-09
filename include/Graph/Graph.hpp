#pragma once
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Layer {
 private:
  int id;
  std::string name;
  std::string type;
  std::string version;
  int numInputs;
  int numNeurons;
  std::vector<std::vector<double>> weights;

 public:
  Layer() {

  }
  Layer(int inputs, int neurons) : numInputs(inputs), numNeurons(neurons) {
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
  double activationFunction(double x) {
    return 1.0 / (1.0 + exp(-x));
  }
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
  }
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
    layers.resize(BiggestSize);
    arrayV.push_back(0);
    V = 0;
  }
  void addEdge(int i, int j) {
    if (i== j) {
      throw std::out_of_range("i=j cant add edge");
    }
    for (int ind = 1; ind < arrayV.size()-i-1;ind++)
     arrayV[i + ind]++;
    arrayE.insert(arrayE.begin() + arrayV[i], j);
    arrayV[V] = arrayE.size();
  }
  void addLayer(Layer lay) {
    layers.push_back(lay);
    if (V == 0) {
      arrayV.push_back(0);
    }
    else {
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
  void checkarrays() { 
    for (size_t i = 0; i < arrayV.size()-1; ++i) {
    std::cout << arrayV[i] << " ";
    }
    std::cout << " " << arrayV[arrayV.size()-1];
    std::cout << "\n";
    for (size_t i = 0; i < arrayE.size(); ++i) {
    std::cout << arrayE[i] << " ";
    }
    std::cout << "\n";
  }
};
