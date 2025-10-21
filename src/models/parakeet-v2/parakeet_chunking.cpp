#include "eddy/models/parakeet-v2/parakeet_chunking.hpp"
#include "eddy/models/parakeet-v2/detail/parakeet_impl.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>

namespace eddy::parakeet {

namespace {

// Named constants for deduplication and chunking
constexpr size_t DEFAULT_DEDUP_WINDOW = 15;
constexpr size_t DEFAULT_BOUNDARY_SEARCH_FRAMES = 20;  // ~1.6s at 12.5 fps (encoder outputs 1 frame per 0.08s)
constexpr size_t DEFAULT_MAX_OVERLAP_TOKENS = 30;

// Cached environment variable checks (computed once per program execution)

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

// Number of frames to search within for boundary-limited overlap detection.
// Default 20; override with EDDY_BOUNDARY_SEARCH_FRAMES (positive integer).
size_t boundary_search_frames() {
  static size_t cached = []() {
    if (const char* e = std::getenv("EDDY_BOUNDARY_SEARCH_FRAMES")) {
      try {
        int v = std::stoi(e);
        if (v > 0 && v < 10000) return static_cast<size_t>(v);
      } catch (const std::exception& ex) {
        std::cerr << "[WARN] Invalid EDDY_BOUNDARY_SEARCH_FRAMES value '" << e << "', using default\n";
      }
    }
    return DEFAULT_BOUNDARY_SEARCH_FRAMES;
  }();
  return cached;
}

// Maximum number of tokens to consider in overlap detection.
// Default 30; override with EDDY_MAX_OVERLAP_TOKENS (positive integer).
size_t max_overlap_tokens() {
  static size_t cached = []() {
    if (const char* e = std::getenv("EDDY_MAX_OVERLAP_TOKENS")) {
      try {
        int v = std::stoi(e);
        if (v > 0 && v < 10000) return static_cast<size_t>(v);
      } catch (const std::exception& ex) {
        std::cerr << "[WARN] Invalid EDDY_MAX_OVERLAP_TOKENS value '" << e << "', using default\n";
      }
    }
    return DEFAULT_MAX_OVERLAP_TOKENS;
  }();
  return cached;
}

// Whether holdback is disabled (set EDDY_DISABLE_HOLDBACK=1 to disable).
bool is_holdback_disabled() {
  static bool cached = []() {
    if (const char* e = std::getenv("EDDY_DISABLE_HOLDBACK")) {
      return std::strcmp(e, "1") == 0;
    }
    return false;
  }();
  return cached;
}

// Number of frames to hold back for right-context lookahead.
// Returns 0 if holdback disabled; override with EDDY_HOLDBACK_FRAMES (positive integer).
// Returns SIZE_MAX if using default (overlap_frames).
size_t holdback_frames_override() {
  static size_t cached = []() {
    if (const char* e = std::getenv("EDDY_HOLDBACK_FRAMES")) {
      try {
        int v = std::stoi(e);
        if (v <= 0) return size_t(0);  // Disable holdback
        if (v < 10000) return static_cast<size_t>(v);
      } catch (const std::exception& ex) {
        std::cerr << "[WARN] Invalid EDDY_HOLDBACK_FRAMES value '" << e << "', using default\n";
      }
    }
    return SIZE_MAX;  // Sentinel: use default
  }();
  return cached;
}

// Whether debug logging is enabled (set EDDY_DEBUG=1 to enable).
bool is_debug_enabled() {
  static bool cached = (std::getenv("EDDY_DEBUG") != nullptr);
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
  // Cache debug flag to avoid repeated getenv calls
  const bool debug = is_debug_enabled();

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
      if (debug) {
        std::cerr << "[EDDY_DEBUG] Time-gate skipped " << time_gate_idx
                  << " tokens to enforce monotonic global timing\n";
      }
      skip_count = std::max(skip_count, time_gate_idx);
    }
  }

  if (debug) {
    std::cerr << "[EDDY_DEBUG] Last 10 tokens of prev chunk: ";
    for (size_t i = std::max(size_t(0), prev_tokens.size() - 10); i < prev_tokens.size(); ++i) {
      std::cerr << prev_tokens[i] << " ";
    }
    std::cerr << "\n[EDDY_DEBUG] First 10 tokens of curr chunk: ";
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
    } catch (const std::exception& ex) {
      if (debug) {
        std::cerr << "[EDDY_DEBUG] is_punctuation failed: " << ex.what() << "\n";
      }
      is_punc = false;
    }
    if (last_prev == first_curr && is_punc) {
      punctuation_removed = 1;
      result.punctuation_removed = 1;
      if (debug) std::cerr << "[EDDY_DEBUG] Dropping duplicate leading punctuation at chunk start\n";
    }
  }

  // Parameters (cached from environment variables)
  const size_t boundary_search = boundary_search_frames();
  const size_t max_overlap = max_overlap_tokens();

  // 2) Exact suffix-prefix match (longest-first)
  size_t prev_tail_window = std::min(prev_tail_window_tokens(), prev_tokens.size());
  size_t working_curr_count = curr_tokens.size() >= punctuation_removed ? (curr_tokens.size() - punctuation_removed) : 0;
  size_t max_match_len = std::min({prev_tail_window, max_overlap, working_curr_count});

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
      if (debug) std::cerr << "[EDDY_DEBUG] Exact suffix-prefix overlap length " << overlap_len << "\n";
      break;
    }
  }

  size_t dedup_skip = punctuation_removed;
  if (best_exact_overlap > 0) {
    dedup_skip += best_exact_overlap;
  } else {
    // 3) Boundary-limited partial overlap search
    // Complexity: O(max_overlap × prev_tail_window × search_limit_tokens)
    // Worst case: O(30 × 15 × 20) = ~9000 iterations per chunk
    // Limit currentStart by frames within the right-context boundary window
    size_t search_limit_tokens = 0;
    if (!curr_timings.empty()) {
      for (size_t idx = 0; idx < curr_timings.size(); ++idx) {
        const size_t local_frame = curr_timings[idx].frame_index - offset;  // convert to local
        if (local_frame >= boundary_search) break;
        // Only count tokens after punctuation_removed
        if (idx >= punctuation_removed) search_limit_tokens = (idx - punctuation_removed) + 1;
      }
    } else {
      // Fallback: use token count if timings absent
      search_limit_tokens = std::min(working_curr_count, boundary_search);
    }

    // Search for any prev subsequence against early part of current
    size_t effective_prev_tail = std::min(prev_tail_window_tokens(), prev_tokens.size());
    for (size_t overlap_len = std::min({effective_prev_tail, max_overlap, working_curr_count});
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
            if (debug) {
              std::cerr << "[EDDY_DEBUG] Boundary duplicate seq len=" << overlap_len
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
    const bool disable_holdback = is_holdback_disabled();
    const size_t holdback_override = holdback_frames_override();

    size_t right_context_frames;
    if (holdback_override == 0) {
      // Holdback explicitly disabled
      right_context_frames = 0;
    } else if (holdback_override == SIZE_MAX) {
      // Use default: min(overlap_frames, chunk_size)
      right_context_frames = std::min(overlap_frames, chunk_size);
    } else {
      // Use override value, capped to chunk_size
      right_context_frames = std::min(holdback_override, chunk_size);
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
        if (debug) {
          std::cerr << "[EDDY_DEBUG] Holding back " << (curr_tokens.size() - holdback_start)
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
