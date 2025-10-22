// Simple SentencePiece-like tokenizer utilities for Parakeet
// Declared for reuse across modules and unit tests.

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace eddy::parakeet {

class Tokenizer {
public:
  void load(const std::string& path, int blank_id);

  std::string decode(const std::vector<int>& token_ids) const;
  std::string decode_span(const int* tokens, size_t count) const;

  size_t vocab_size() const;
  int blank_id() const;
  bool is_punctuation(int token_id) const;

private:
  std::vector<std::string> vocab_;
  int blank_id_ = 1024;
};

}  // namespace eddy::parakeet

