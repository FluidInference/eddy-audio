#include "utils/openvino_utils_detail.hpp"

#include <openvino/openvino.hpp>
#include <openvino/op/log_softmax.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/relu.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

bool expect(bool condition, std::string_view message) {
  if (condition) return true;
  std::cerr << "FAILED: " << message << "\n";
  return false;
}

std::shared_ptr<ov::Model> make_log_softmax_model(const ov::PartialShape& shape,
                                                  int64_t axis,
                                                  std::shared_ptr<ov::op::v5::LogSoftmax>& op) {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
  op = std::make_shared<ov::op::v5::LogSoftmax>(input, axis);
  return std::make_shared<ov::Model>(ov::OutputVector{op}, ov::ParameterVector{input});
}

std::vector<float> infer_cpu(const std::shared_ptr<ov::Model>& model,
                             const ov::Shape& shape,
                             const std::vector<float>& values) {
  ov::Core core;
  auto compiled = core.compile_model(model, "CPU");
  auto request = compiled.create_infer_request();
  ov::Tensor input(ov::element::f32, shape);
  std::copy(values.begin(), values.end(), input.data<float>());
  request.set_input_tensor(input);
  request.infer();

  const auto output = request.get_output_tensor();
  return {output.data<const float>(), output.data<const float>() + output.get_size()};
}

bool test_last_axis_normalization_preserves_output() {
  const ov::Shape shape{1, 2, 2, 3};
  std::shared_ptr<ov::op::v5::LogSoftmax> op;
  auto model = make_log_softmax_model(shape, -1, op);

  const std::vector<float> values{
      -1.0f, 0.0f, 1.0f, 2.0f, -2.0f, 0.5f,
      4.0f, 3.0f, 2.0f, -3.0f, 1.0f, 5.0f,
  };
  const auto before = infer_cpu(model, shape, values);

  const bool changed = eddy::parakeet::detail::normalize_negative_log_softmax_axes(*model);
  const auto after = infer_cpu(model, shape, values);

  bool ok = expect(changed, "axis -1 reports a change");
  ok &= expect(op->get_axis() == 3, "rank-4 axis -1 normalizes to 3");
  ok &= expect(before.size() == after.size(), "output sizes match");
  for (size_t i = 0; i < before.size() && i < after.size(); ++i) {
    ok &= expect(std::abs(before[i] - after[i]) < 1e-6f,
                 "normalization preserves LogSoftmax output");
  }
  return ok;
}

bool test_first_axis_normalization() {
  std::shared_ptr<ov::op::v5::LogSoftmax> op;
  auto model = make_log_softmax_model(ov::Shape{1, 2, 3, 4}, -4, op);
  const bool changed = eddy::parakeet::detail::normalize_negative_log_softmax_axes(*model);
  return expect(changed, "axis -4 reports a change") &&
         expect(op->get_axis() == 0, "rank-4 axis -4 normalizes to 0");
}

bool test_positive_axis_is_unchanged() {
  std::shared_ptr<ov::op::v5::LogSoftmax> op;
  auto model = make_log_softmax_model(ov::Shape{1, 2, 3, 4}, 2, op);
  const bool changed = eddy::parakeet::detail::normalize_negative_log_softmax_axes(*model);
  return expect(!changed, "positive axis reports no change") &&
         expect(op->get_axis() == 2, "positive axis remains unchanged");
}

bool test_dynamic_rank_is_unchanged() {
  std::shared_ptr<ov::op::v5::LogSoftmax> op;
  auto model = make_log_softmax_model(ov::PartialShape::dynamic(), -1, op);
  const bool changed = eddy::parakeet::detail::normalize_negative_log_softmax_axes(*model);
  return expect(!changed, "dynamic-rank axis reports no change") &&
         expect(op->get_axis() == -1, "dynamic-rank axis remains negative");
}

bool test_non_log_softmax_is_untouched() {
  auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
  auto relu = std::make_shared<ov::op::v0::Relu>(input);
  ov::Model model(ov::OutputVector{relu}, ov::ParameterVector{input});
  return expect(!eddy::parakeet::detail::normalize_negative_log_softmax_axes(model),
                "model without LogSoftmax reports no change");
}

}  // namespace

int main() {
  bool ok = true;
  ok &= test_last_axis_normalization_preserves_output();
  ok &= test_first_axis_normalization();
  ok &= test_positive_axis_is_unchanged();
  ok &= test_dynamic_rank_is_unchanged();
  ok &= test_non_log_softmax_is_untouched();
  return ok ? 0 : 1;
}
