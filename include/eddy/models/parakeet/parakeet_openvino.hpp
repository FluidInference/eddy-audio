#pragma once

#include <memory>
#include <string>

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/models/parakeet/parakeet.hpp"

namespace ov {
class infer_request;
class CompiledModel;
}

namespace eddy::parakeet {

// Forward declaration
class OpenVINOParakeet;

std::shared_ptr<OpenVINOParakeet> make_openvino_parakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                                                        ModelPaths model_paths,
                                                        RuntimeConfig runtime_cfg);

class OpenVINOParakeet : public IParakeetModel {
public:
  OpenVINOParakeet(std::shared_ptr<eddy::OpenVINOBackend> backend,
                   ModelPaths model_paths,
                   RuntimeConfig runtime_cfg);
  ~OpenVINOParakeet() override;

  InferenceResult infer(const AudioSegment& segment, const SegmentOptions& options) override;

  void warmup();

  // Public to allow helper functions in implementation file
  struct Impl;

private:
  void ensure_compiled_model();

  std::unique_ptr<Impl> impl_;
};

}  // namespace eddy::parakeet
