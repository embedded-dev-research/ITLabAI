#include <vector>

#include "layers/PoolingLayer.hpp"

TEST(poolinglayer, test1) {
  Shape inp = {4, 4};
  Shape pool = {2, 2};
  PoolingLayer<double> a = PoolingLayer<double>(inp, pool, "average");
  std::vector<double> output = a.run(
      std::vector<double>({9, 8, 7, 6, 5, 4, 3, 2, 2, 3, 4, 5, 6, 7, 8, 9}));
  ASSERT_NO_THROW(true);
}
