#pragma once

#include <openvino/core/model.hpp>

namespace eddy::parakeet::detail {

// Canonicalize valid negative opset5 LogSoftmax axes when the input rank is
// static. Returns true when at least one node changed.
bool normalize_negative_log_softmax_axes(ov::Model& model);

}  // namespace eddy::parakeet::detail
