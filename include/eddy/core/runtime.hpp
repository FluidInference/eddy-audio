#pragma once

#include <memory>

namespace eddy {

class IBackend;

class Runtime {
public:
  Runtime();
  explicit Runtime(std::shared_ptr<IBackend> backend);

  std::shared_ptr<IBackend> backend() const;
  void set_backend(std::shared_ptr<IBackend> backend);

private:
  std::shared_ptr<IBackend> backend_;
};

}  // namespace eddy
