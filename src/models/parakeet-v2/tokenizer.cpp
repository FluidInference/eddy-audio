#include "eddy/models/parakeet-v2/tokenizer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace eddy::parakeet {

// SentencePiece word boundary marker (U+2581 "▁" Lower One Eighth Block)
// This character appears at the start of words in the tokenized output
static constexpr std::string_view kWordBoundary = "▁";

void Tokenizer::load(const std::string& path, int blank_id) {
  std::ifstream stream(path);
  if (!stream.good()) {
    throw std::runtime_error("Failed to open Parakeet vocabulary: " + path);
  }

  nlohmann::json json;
  stream >> json;

  // Handle two vocabulary formats:
  // V2 format: {"0": "token0", "1": "token1", ...}
  // V3 format: {"id_to_token": {"0": "token0", "1": "token1", ...}, "vocab_size": 8192, ...}
  nlohmann::json vocab_json;
  if (json.contains("id_to_token")) {
    // V3 format with nested id_to_token
    vocab_json = json["id_to_token"];
  } else {
    // V2 format (flat dictionary)
    vocab_json = json;
  }

  // Find max token ID to size vocabulary array
  size_t max_id = 0;
  for (const auto& item : vocab_json.items()) {
    const size_t id = static_cast<size_t>(std::stoul(item.key()));
    max_id = std::max(max_id, id);
  }

  // Populate vocabulary from JSON
  vocab_.assign(max_id + 1, std::string{});
  for (const auto& item : vocab_json.items()) {
    const size_t id = static_cast<size_t>(std::stoul(item.key()));
    vocab_[id] = item.value().get<std::string>();
  }

  // Ensure blank token ID is within vocabulary range
  blank_id_ = blank_id;
  if (blank_id_ >= static_cast<int>(vocab_.size())) {
    vocab_.resize(static_cast<size_t>(blank_id_) + 1U);
  }
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
  return decode_span(token_ids.data(), token_ids.size());
}

std::string Tokenizer::decode_span(const int* tokens, size_t count) const {
  std::string result;
  bool first_piece = true;

  for (size_t idx_t = 0; idx_t < count; ++idx_t) {
    int token_id = tokens[idx_t];

    // Skip blank tokens and invalid IDs
    if (token_id == blank_id_ || token_id < 0) {
      continue;
    }

    const auto idx = static_cast<size_t>(token_id);
    if (idx >= vocab_.size()) {
      continue;
    }

    // Get the token piece from vocabulary
    std::string_view piece{vocab_[idx]};
    if (piece.empty()) {
      continue;
    }

    // Handle SentencePiece word boundary marker
    bool prepend_space = piece.starts_with(kWordBoundary);
    if (prepend_space) {
      piece.remove_prefix(kWordBoundary.size());
    }

    // Append to result with appropriate spacing
    if (!piece.empty()) {
      if (!first_piece && prepend_space && !result.empty()) {
        result.push_back(' ');
      }
      result.append(piece);
      first_piece = false;
    }
  }

  return result;
}

size_t Tokenizer::vocab_size() const { return vocab_.size(); }
int Tokenizer::blank_id() const { return blank_id_; }

bool Tokenizer::is_punctuation(int token_id) const {
  if (token_id < 0) {
    return false;
  }

  const auto idx = static_cast<size_t>(token_id);
  if (idx >= vocab_.size()) {
    return false;
  }

  std::string_view piece{vocab_[idx]};
  if (piece.empty()) {
    return false;
  }

  // Remove SentencePiece word boundary marker if present
  if (piece.starts_with(kWordBoundary)) {
    piece.remove_prefix(kWordBoundary.size());
  }

  return (piece == "." || piece == "?" || piece == "!");
}

}  // namespace eddy::parakeet

