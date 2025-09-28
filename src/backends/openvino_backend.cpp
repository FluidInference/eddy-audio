#include "eddy/backends/openvino_backend.hpp"

#include <openvino/openvino.hpp>

#include <memory>
#include <stdexcept>
#include <utility>

namespace eddy {

struct OpenVINOBackend::Impl {
  OpenVINOOptions options;
  std::shared_ptr<ov::Core> core;
};

OpenVINOBackend::OpenVINOBackend()
    : OpenVINOBackend(OpenVINOOptions{}) {}

OpenVINOBackend::OpenVINOBackend(OpenVINOOptions options)
    : impl_(std::make_unique<Impl>()) {
  impl_->options = std::move(options);
  impl_->core = std::make_shared<ov::Core>();
  if (!impl_->options.cache_dir.empty()) {
    impl_->core->set_property(ov::cache_dir(impl_->options.cache_dir));
  }
}

OpenVINOBackend::~OpenVINOBackend() = default;

std::string_view OpenVINOBackend::id() const {
  return "openvino";
}

std::shared_ptr<Model> OpenVINOBackend::load_model(const LoadModelParams& params) {
  (void)params;
  throw std::runtime_error("Generic OpenVINO load_model not implemented; use model-specific loaders.");
}

ov::Core& OpenVINOBackend::core() {
  return *impl_->core;
}

const ov::Core& OpenVINOBackend::core() const {
  return *impl_->core;
}

const OpenVINOOptions& OpenVINOBackend::options() const {
  return impl_->options;
}

}  // namespace eddy
