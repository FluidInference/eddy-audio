#pragma once

#include <memory>
#include <string>

#include "eddy/backends/backend.hpp"

namespace ov {
class Core;
}

namespace eddy {

struct OpenVINOOptions {
  std::string device = "AUTO";
  std::string cache_dir;
};

class OpenVINOBackend : public IBackend {
public:
  OpenVINOBackend();
  explicit OpenVINOBackend(OpenVINOOptions options);
  ~OpenVINOBackend() override;

  std::string_view id() const override;
  std::shared_ptr<Model> load_model(const LoadModelParams& params) override;

  ov::Core& core();
  const ov::Core& core() const;
  const OpenVINOOptions& options() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace eddy
