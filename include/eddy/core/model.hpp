#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace eddy {

struct TensorInfo {
  std::string name;
  std::vector<size_t> shape;
  std::string dtype;
};

class Model {
public:
  virtual ~Model() = default;
};

}  // namespace eddy
