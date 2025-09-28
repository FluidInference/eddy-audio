#pragma once

#include <memory>
#include <string>
#include <string_view>

namespace eddy {

class Model;

struct LoadModelParams {
  std::string uri;
  std::string device;
};

class IBackend {
public:
  virtual ~IBackend() = default;
  virtual std::string_view id() const = 0;
  virtual std::shared_ptr<Model> load_model(const LoadModelParams& params) = 0;
};

}  // namespace eddy
