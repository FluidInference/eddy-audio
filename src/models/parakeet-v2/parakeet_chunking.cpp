#include "eddy/models/parakeet-v2/parakeet_chunking.hpp"
#include "parakeet_openvino_impl.hpp"

#include <algorithm>
#include <iostream>
#include <string>

namespace eddy::parakeet {

namespace {

// Named constants for deduplication and chunking
constexpr size_t DEFAULT_DEDUP_WINDOW = 15;
constexpr size_t DEFAULT_BOUNDARY_SEARCH_FRAMES = 20;  // ~1.6s at 12.5 fps
constexpr size_t DEFAULT_MAX_OVERLAP_TOKENS = 30;

// Number of previous tokens to consider for exact/partial overlap checks in dedup.
// Default 15; override with EDDY_DEDUP_PREV_TOKENS (positive integer).
size_t prev_tail_window_tokens() {
  static size_t cached = []() {
    size_t def = DEFAULT_DEDUP_WINDOW;
    if (const char* e = std::getenv("EDDY_DEDUP_PREV_TOKENS")) {
      try {
        int v = std::stoi(e);
        if (v > 0 && v < 10000) return static_cast<size_t>(v);
      } catch (const std::exception& ex) {
        std::cerr << "[WARN] Invalid EDDY_DEDUP_PREV_TOKENS value '" << e << "', using default\n";
      }
    }
    return def;
  }();
  return cached;
}

}  // namespace

ChunkDeduplicationResult deduplicate_chunk(
    ParakeetImpl& impl,
    const std::vector<int>& prev_tokens,
    const std::vector<int>& curr_tokens,
    std::vector<TokenTiming>& curr_timings,
    size_t offset,
    size_t chunk_size,
    size_t overlap_frames,
    bool is_last_chunk,
    size_t last_emitted_global_frame,
    bool have_last_emitted_frame
) {
  ChunkDeduplicationResult result;
  result.emit_end = curr_tokens.size();  // Default: emit everything

  // Convert timings to global frame indices by adding the current chunk offset
  // Check for overflow before adding offset
  for (auto& t : curr_timings) {
    if (t.frame_index > SIZE_MAX - offset) {
      throw std::runtime_error("Integer overflow in frame_index adjustment: frame_index=" +
                               std::to_string(t.frame_index) + " offset=" + std::to_string(offset));
    }
    t.frame_index += offset;
  }

  size_t skip_count = 0;

  // Global time-gate: ensure strictly increasing global frame indices
  if (have_last_emitted_frame && !curr_timings.empty()) {
    size_t time_gate_idx = 0;
    while (time_gate_idx < curr_timings.size() &&
           curr_timings[time_gate_idx].frame_index <= last_emitted_global_frame) {
      ++time_gate_idx;
    }
    if (time_gate_idx > 0) {
      if (std::getenv("EDDY_DEBUG")) {
        std::cerr << "[INFO] Time-gate skipped " << time_gate_idx
                  << " tokens to enforce monotonic global timing\n";
      }
      skip_count = std::max(skip_count, time_gate_idx);
    }
  }

  if (std::getenv("EDDY_DEBUG")) {
    std::cerr << "[DEBUG] Last 10 tokens of prev chunk: ";
    for (size_t i = std::max(size_t(0), prev_tokens.size() - 10); i < prev_tokens.size(); ++i) {
      std::cerr << prev_tokens[i] << " ";
    }
    std::cerr << "\n[DEBUG] First 10 tokens of curr chunk: ";
    for (size_t i = 0; i < std::min(size_t(10), curr_tokens.size()); ++i) {
      std::cerr << curr_tokens[i] << " ";
    }
    std::cerr << "\n";
  }

  // FluidAudio-style deduplication:
  // 1) Punctuation guard
  // 2) Exact suffix-prefix overlap (longest-first)
  // 3) Boundary-limited partial overlap search within the beginning of the current chunk

  // 1) Punctuation guard: if previous tail token equals current head and is punctuation, drop the head
  size_t punctuation_removed = 0;
  if (!prev_tokens.empty() && !curr_tokens.empty()) {
    int last_prev = prev_tokens.back();
    int first_curr = curr_tokens.front();
    bool is_punc = false;
    try {
      is_punc = impl.tokenizer.is_punctuation(first_curr);
    } catch (...) {
      is_punc = false;
    }
    if (last_prev == first_curr && is_punc) {
      punctuation_removed = 1;
      result.punctuation_removed = 1;
      if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Dropping duplicate leading punctuation at chunk start\n";
    }
  }

  // Parameters (tunable via env):
  size_t boundary_search_frames = DEFAULT_BOUNDARY_SEARCH_FRAMES;
  if (const char* env_b = std::getenv("EDDY_BOUNDARY_SEARCH_FRAMES")) {
    try {
      int v = std::stoi(env_b);
      if (v > 0) boundary_search_frames = static_cast<size_t>(v);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] Invalid EDDY_BOUNDARY_SEARCH_FRAMES value '" << env_b << "', using default\n";
    }
  }
  size_t max_overlap_tokens = DEFAULT_MAX_OVERLAP_TOKENS;
  if (const char* env_o = std::getenv("EDDY_MAX_OVERLAP_TOKENS")) {
    try {
      int v = std::stoi(env_o);
      if (v > 0) max_overlap_tokens = static_cast<size_t>(v);
    } catch (const std::exception& e) {
      std::cerr << "[WARN] Invalid EDDY_MAX_OVERLAP_TOKENS value '" << env_o << "', using default\n";
    }
  }

  // 2) Exact suffix-prefix match (longest-first)
  size_t prev_tail_window = std::min(prev_tail_window_tokens(), prev_tokens.size());
  size_t working_curr_count = curr_tokens.size() >= punctuation_removed ? (curr_tokens.size() - punctuation_removed) : 0;
  size_t max_match_len = std::min({prev_tail_window, max_overlap_tokens, working_curr_count});

  size_t best_exact_overlap = 0;
  for (size_t overlap_len = max_match_len; overlap_len > 1; --overlap_len) {
    bool match = true;
    const size_t prev_start = prev_tokens.size() - overlap_len;
    for (size_t i = 0; i < overlap_len; ++i) {
      if (prev_tokens[prev_start + i] != curr_tokens[punctuation_removed + i]) {
        match = false;
        break;
      }
    }
    if (match) {
      best_exact_overlap = overlap_len;
      result.exact_overlap = overlap_len;
      if (std::getenv("EDDY_DEBUG")) std::cerr << "[INFO] Exact suffix-prefix overlap length " << overlap_len << "\n";
      break;
    }
  }

  size_t dedup_skip = punctuation_removed;
  if (best_exact_overlap > 0) {
    dedup_skip += best_exact_overlap;
  } else {
    // 3) Boundary-limited partial overlap search
    // Limit currentStart by frames within the right-context boundary window
    size_t search_limit_tokens = 0;
    if (!curr_timings.empty()) {
      for (size_t idx = 0; idx < curr_timings.size(); ++idx) {
        const size_t local_frame = curr_timings[idx].frame_index - offset;  // convert to local
        if (local_frame >= boundary_search_frames) break;
        // Only count tokens after punctuation_removed
        if (idx >= punctuation_removed) search_limit_tokens = (idx - punctuation_removed) + 1;
      }
    } else {
      // Fallback: use token count if timings absent
      search_limit_tokens = std::min(working_curr_count, boundary_search_frames);
    }

    // Search for any prev subsequence against early part of current
    size_t effective_prev_tail = std::min(prev_tail_window_tokens(), prev_tokens.size());
    for (size_t overlap_len = std::min({effective_prev_tail, max_overlap_tokens, working_curr_count});
         overlap_len > 1; --overlap_len) {
      const size_t prev_start_min = prev_tokens.size() > effective_prev_tail ? (prev_tokens.size() - effective_prev_tail) : 0;
      const size_t prev_end = prev_tokens.size() >= overlap_len ? (prev_tokens.size() - overlap_len + 1) : 0;
      if (prev_end <= prev_start_min) continue;

      for (size_t prev_start = prev_start_min; prev_start < prev_end; ++prev_start) {
        // Iterate currentStart within boundary window
        const size_t curr_end_limit = (working_curr_count >= overlap_len) ? (working_curr_count - overlap_len + 1) : 0;
        const size_t search_limit = std::min(search_limit_tokens, curr_end_limit);
        for (size_t curr_off = 0; curr_off < search_limit; ++curr_off) {
          bool eq = true;
          for (size_t k = 0; k < overlap_len; ++k) {
            if (prev_tokens[prev_start + k] != curr_tokens[punctuation_removed + curr_off + k]) { eq = false; break; }
          }
          if (eq) {
            dedup_skip = std::max(dedup_skip, punctuation_removed + curr_off + overlap_len);
            result.boundary_overlap = overlap_len;
            if (std::getenv("EDDY_DEBUG")) {
              std::cerr << "[INFO] Boundary duplicate seq len=" << overlap_len
                        << ", curr_off=" << curr_off << ", prev_start=" << prev_start << "\n";
            }
            // Break out of loops in order: curr_off, prev_start
            curr_off = search_limit;  // force exit
            prev_start = prev_end;    // force exit
            overlap_len = 1;          // set to loop exit threshold (>1 condition)
            break;
          }
        }
      }
    }
  }

  if (dedup_skip > 0) {
    skip_count = std::max(skip_count, dedup_skip);
  }

  // Right-context holdback: for non-final chunks, optionally hold back tokens
  // near the end so that the next chunk (with more right context) can decide.
  size_t emit_end = curr_tokens.size();
  if (!is_last_chunk && !curr_timings.empty()) {
    bool disable_holdback = false;
    if (const char* env_hb = std::getenv("EDDY_DISABLE_HOLDBACK")) {
      disable_holdback = std::string(env_hb) == "1";
    }
    size_t right_context_frames = std::min(overlap_frames, chunk_size);
    if (const char* env_hbf = std::getenv("EDDY_HOLDBACK_FRAMES")) {
      try {
        int v = std::stoi(env_hbf);
        if (v <= 0) {
          disable_holdback = true;
        } else {
          right_context_frames = std::min<size_t>(static_cast<size_t>(v), chunk_size);
        }
      } catch (const std::exception& e) {
        std::cerr << "[WARN] Invalid EDDY_HOLDBACK_FRAMES value '" << env_hbf << "', using default\n";
      }
    }
    if (!disable_holdback && right_context_frames > 0) {
      size_t holdback_start = curr_tokens.size();
      for (size_t idx = 0; idx < curr_timings.size(); ++idx) {
        const size_t local_frame = curr_timings[idx].frame_index - offset;  // convert back to local
        if (local_frame >= (chunk_size > right_context_frames ? (chunk_size - right_context_frames) : 0)) {
          holdback_start = idx;
          break;
        }
      }
      if (holdback_start < curr_tokens.size()) {
        emit_end = std::min(emit_end, holdback_start);
        result.holdback_count = curr_tokens.size() - holdback_start;
        if (std::getenv("EDDY_DEBUG")) {
          std::cerr << "[INFO] Holding back " << (curr_tokens.size() - holdback_start)
                    << " tokens for right-context lookahead\n";
        }
      }
    }
  }

  result.skip_prefix = skip_count;
  result.emit_end = emit_end;
  return result;
}

}  // namespace eddy::parakeet
