#pragma once

#include <dnnl.hpp>
#include <memory>
#include <vector>

#include "layers/Layer.hpp"
#include "layers/Tensor.hpp"

namespace it_lab_ai {

class ConcatLayerOneDnn : public Layer {
 public:
  explicit ConcatLayerOneDnn(int64_t axis = 0) : Layer(kConcat), axis_(axis) {}

  ConcatLayerOneDnn(const ConcatLayerOneDnn& c)
      : Layer(kConcat), axis_(c.axis_) {}

  void run(const std::vector<Tensor>& input,
           std::vector<Tensor>& output) override;

#ifdef ENABLE_STATISTIC_WEIGHTS
  Tensor get_weights() override {
    std::vector<int> v = {0};
    return make_tensor(v);
  }
#endif

 private:
  int64_t axis_;

  bool initialized_ = false;
  Type last_type_;
  std::vector<Shape> last_shapes_;

  std::unique_ptr<dnnl::engine> engine_;
  std::unique_ptr<dnnl::stream> stream_;
  std::unique_ptr<dnnl::concat> concat_prim_;

  std::vector<dnnl::memory::desc> src_mds_;
  dnnl::memory::desc dst_md_;

  Shape output_shape_;

  std::vector<dnnl::memory> src_mems_;
  dnnl::memory dst_mem_;
  std::unordered_map<int, dnnl::memory> args_;

  std::vector<float> dst_buffer_f32_;
  std::vector<int> dst_buffer_s32_;

  void initialize_onednn(const std::vector<Tensor>& input);

  static void validate_input(const std::vector<Tensor>& input);

  [[nodiscard]] static dnnl::memory::data_type get_dnnl_data_type(Type type);

  [[nodiscard]] static dnnl::memory::format_tag pick_format(size_t ndims);

  [[nodiscard]] static std::vector<dnnl::memory::dim> shape_to_dims(
      const Shape& shape);

  [[nodiscard]] static Shape calculate_output_shape(
      const std::vector<Tensor>& inputs, int64_t axis);

  [[nodiscard]] static int64_t normalize_axis(int64_t axis, size_t rank);
};

}  // namespace it_lab_ai
