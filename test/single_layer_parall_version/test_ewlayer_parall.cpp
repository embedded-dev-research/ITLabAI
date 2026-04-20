#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "layers/EWLayer.hpp"

#define ENABLE_TIMING_OUTPUT 1

#if ENABLE_TIMING_OUTPUT
#  define PRINT_TIMING(msg) std::cout << msg << std::endl
#else
#  define PRINT_TIMING(msg) ((void)0)
#endif

using namespace it_lab_ai;

TEST(ewlayer_parall, parallel_for_ew_relu) {
  EWLayer layer("relu");

  std::vector<int> vec(8000000, -1);
  Tensor input = make_tensor<int>(vec);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING(" time: " << duration.count() << " ms");
    for (size_t i = 0; i < 8000000; i++) {
      EXPECT_EQ((*out[0].as<int>())[i], 0);
    }
  }
}

TEST(ewlayer_parall, parallel_for_sigmoid) {
  EWLayer layer("sigmoid");

  std::vector<int> vec(8000000, -1);
  Tensor input = make_tensor<int>(vec);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING(" time: " << duration.count() << " ms");
    for (size_t i = 0; i < 8000000; i++) {
      EXPECT_EQ((*out[0].as<int>())[i], 0);
    }
  }
}

TEST(ewlayer_parall, parallel_for_minus) {
  EWLayer layer("minus");

  std::vector<int> vec(8000000, -1);
  Tensor input = make_tensor<int>(vec);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING(" time: " << duration.count() << " ms");
    for (size_t i = 0; i < 8000000; i++) {
      EXPECT_EQ((*out[0].as<int>())[i], 1);
    }
  }
}

TEST(ewlayer_parall, parallel_for_linear) {
  EWLayer layer("linear", 2.0F, 2.0F);

  std::vector<int> vec(8000000, -1);
  Tensor input = make_tensor<int>(vec);
  Tensor output;
  std::vector<Tensor> in{input};
  std::vector<Tensor> out{output};

  std::vector<ParBackend> backends = {ParBackend::kSeq, ParBackend::kThreads,
                                      ParBackend::kTbb, ParBackend::kOmp,
                                      ParBackend::kKokkos};

  for (auto backend : backends) {
    RuntimeOptions options;
    options.par_backend = backend;

    auto start = std::chrono::high_resolution_clock::now();
    layer.run(in, out, options);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    PRINT_TIMING(" time: " << duration.count() << " ms");
    for (size_t i = 0; i < 8000000; i++) {
      EXPECT_EQ((*out[0].as<int>())[i], 0);
    }
  }
}

TEST(ewlayer_parall, parallel_for_direct) {
  const int SIZE = 2000;
  std::vector<int> matrix1(SIZE * SIZE);
  std::vector<int> matrix2(SIZE * SIZE);
  std::vector<int> result(SIZE * SIZE);

  for (int i = 0; i < SIZE * SIZE; ++i) {
    matrix1[i] = 1;
    matrix2[i] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kSeq);

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  PRINT_TIMING(" time: " << total_duration.count() << " ms");

  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kThreads);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kTbb);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kOmp);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }
  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + matrix2[i];
  }, ParBackend::kKokkos);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }
}

TEST(ewlayer_parall, parallel_for_notmatrix) {
  const int SIZE = 3000;
  std::vector<int> matrix1(SIZE * SIZE);
  std::vector<int> result(SIZE * SIZE);

  for (int i = 0; i < SIZE * SIZE; ++i) {
    matrix1[i] = 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kSeq);

  auto end = std::chrono::high_resolution_clock::now();
  auto total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");

  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kThreads);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kTbb);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kOmp);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }

  start = std::chrono::high_resolution_clock::now();
  parallel::parallel_for(SIZE * SIZE, [&](std::size_t i) {
    result[i] = matrix1[i] + 1;
  }, ParBackend::kKokkos);
  end = std::chrono::high_resolution_clock::now();
  total_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  PRINT_TIMING(" time: " << total_duration.count() << " ms");
  for (int i = 0; i < SIZE * SIZE; i++) {
    ASSERT_EQ(result[i], 2);
  }
}
