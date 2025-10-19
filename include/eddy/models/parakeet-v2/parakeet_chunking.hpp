#pragma once

#include "eddy/models/parakeet-v2/parakeet.hpp"
#include <vector>
#include <cstddef>

namespace eddy::parakeet {

// Forward declarations
struct ParakeetImpl;

// Result of chunk deduplication processing
struct ChunkDeduplicationResult {
  size_t skip_prefix = 0;     // Number of tokens to skip from current chunk start
  size_t emit_end;            // Index where to stop emitting (for holdback)

  // Debug info
  size_t punctuation_removed = 0;
  size_t exact_overlap = 0;
  size_t boundary_overlap = 0;
  size_t holdback_count = 0;
};

// Process chunk deduplication to remove overlaps between consecutive chunks
//
// This implements FluidAudio-style 3-stage deduplication:
// 1. Punctuation guard - remove duplicate punctuation at boundary
// 2. Exact suffix-prefix match - find longest matching sequence
// 3. Boundary-limited partial overlap - search within right-context window
// 4. Right-context holdback - delay tokens near chunk end
//
// Parameters:
//   impl: Model implementation (for tokenizer access)
//   prev_tokens: Previously emitted tokens
//   curr_tokens: Current chunk tokens
//   curr_timings: Current chunk token timings (modified: frame indices adjusted to global)
//   offset: Current chunk offset in global mel frames
//   chunk_size: Size of current chunk in frames
//   overlap_frames: Number of overlapping frames between chunks
//   is_last_chunk: Whether this is the final chunk
//   last_emitted_global_frame: Last emitted token's global frame (for time-gating)
//   have_last_emitted_frame: Whether we have a previous frame reference
//
// Returns:
//   ChunkDeduplicationResult with skip/emit indices
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
);

}  // namespace eddy::parakeet
