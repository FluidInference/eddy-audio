#include "eddy/core/runtime.hpp"

#include "eddy/backends/backend.hpp"

namespace eddy {

Runtime::Runtime() = default;

Runtime::Runtime(std::shared_ptr<IBackend> backend)
    : backend_(std::move(backend)) {}

std::shared_ptr<IBackend> Runtime::backend() const {
  return backend_;
}

void Runtime::set_backend(std::shared_ptr<IBackend> backend) {
  backend_ = std::move(backend);
}

}  // namespace eddy
