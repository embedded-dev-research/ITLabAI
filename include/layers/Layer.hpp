#pragma once
#include <omp.h>

#include <algorithm>
#include <execution>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "layers/Shape.hpp"
#include "layers/Tensor.hpp"
#include "oneapi/tbb.h"

namespace it_lab_ai {

enum LayerType : uint8_t {
  kInput,
  kPooling,
  kNormalization,
  kDropout,
  kElementWise,
  kConvolution,
  kFullyConnected,
  kFlatten,
  kOutput,
  kConcat,
  kSplit,
  kBinaryOp,
  kReduce,
  kTranspose,
  kReshape,
  kSoftmax,
  kMatmul,
  kBatchNormalization
};

enum ImplType : uint8_t { kDefault, kTBB, kSTL };

class Layer;

struct PostOperations {
  std::vector<Layer*> layers;
  unsigned int count = 0;
};

class Layer {
 public:
  Layer() = default;
  Layer(LayerType type) : type_(type) {}
  virtual ~Layer() = default;
  PostOperations postops;
  int getID() const { return id_; }
  void setID(int id) { id_ = id; }
  void setTypeParall(int type) { type_parall_ = type; }
  LayerType getName() const { return type_; }
  virtual void run(const std::vector<Tensor>& input,
                   std::vector<Tensor>& output) = 0;
#ifdef ENABLE_STATISTIC_WEIGHTS
  virtual Tensor get_weights() = 0;
#endif

 protected:
  int id_ = 0;
  LayerType type_;
  int type_parall_;
};

template <typename ValueType>
class LayerImpl {
 public:
  LayerImpl() = default;
  LayerImpl(const Shape& inputShape, const Shape& outputShape)
      : inputShape_(inputShape), outputShape_(outputShape) {}
  LayerImpl(const LayerImpl& c) = default;
  LayerImpl& operator=(const LayerImpl& c) = default;
  virtual std::vector<ValueType> run(
      const std::vector<ValueType>& input) const = 0;
  Shape get_input_shape() const { return inputShape_; }
  Shape get_output_shape() const { return outputShape_; }
  // weights width x height
  std::pair<Shape, Shape> get_dims() const {
    return std::pair<Shape, Shape>(outputShape_, inputShape_);
  }

 protected:
  Shape inputShape_;
  Shape outputShape_;
};

template <typename Func>
inline void parallel_for(int count, Func func, int mode = 0) {
  static bool stl_available = true;
  static bool tbb_available = true;
  static bool omp_available = true;
  const int MIN_CHUNK_SIZE = 1000;
  if (count < MIN_CHUNK_SIZE) {
    mode = 0;
  }

  switch (mode) {
    case 0:  // Sequential
    {
      for (int i = 0; i < count; ++i) {
        func(i);
      }
      break;
    }

    case 1:  // STL
    {
      if (stl_available) {
        try {
          int num_threads =
              static_cast<int>(std::thread::hardware_concurrency());
          if (num_threads == 0) num_threads = 4;

          int min_chunk_size = std::max(1000, count / (num_threads * 4));
          if (count / num_threads < min_chunk_size) {
            num_threads = std::max(1, count / min_chunk_size);
          }

          std::vector<std::thread> threads;
          threads.reserve(num_threads);

          int chunk_size = count / num_threads;
          int remainder = count % num_threads;

          int start = 0;
          for (int t = 0; t < num_threads; ++t) {
            int end = start + chunk_size + (t < remainder ? 1 : 0);
            if (start >= end) break;

            threads.emplace_back([start, end, &func]() {
              for (int i = start; i < end; ++i) {
                func(i);
              }
            });

            start = end;
          }

          for (auto& thread : threads) {
            thread.join();
          }

        } catch (const std::exception& e) {
          std::cout << "Thread execution failed: " << e.what()
                    << ". Falling back to sequential.\n";
          stl_available = false;
          for (int i = 0; i < count; ++i) func(i);
        }
      } else {
        for (int i = 0; i < count; ++i) func(i);
      }
      break;
    }

    case 2:  // Intel TBB
    {
      if (tbb_available) {
        try {
          oneapi::tbb::parallel_for(
              oneapi::tbb::blocked_range<int>(0, count),
              [&](const oneapi::tbb::blocked_range<int>& range) {
                for (int i = range.begin(); i < range.end(); ++i) {
                  func(i);
                }
              },
              oneapi::tbb::auto_partitioner());
        } catch (const std::exception& e) {
          std::cout << "TBB execution failed: " << e.what()
                    << ". Falling back to sequential.\n";
          tbb_available = false;
          for (int i = 0; i < count; ++i) func(i);
        }
      } else {
        for (int i = 0; i < count; ++i) func(i);
      }
      break;
    }

    case 3:  // OpenMP
    {
      if (omp_available) {
        try {
          int num_threads = omp_get_max_threads();

          int chunk_size = std::max(1000, count / (num_threads * 8));

#pragma omp parallel for schedule(static, chunk_size) num_threads(num_threads)
          for (int i = 0; i < count; ++i) {
            func(i);
          }

        } catch (...) {
          std::cout << "OpenMP execution failed. Falling back to sequential.\n";
          omp_available = false;
          for (int i = 0; i < count; ++i) func(i);
        }
      } else {
        for (int i = 0; i < count; ++i) func(i);
      }
      break;
    }

    default:
      for (int i = 0; i < count; ++i) func(i);
  }
}

}  // namespace it_lab_ai
