// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#include "eddy/streaming/buffered_streaming_asr.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace eddy::streaming {

BufferedStreamingASR::BufferedStreamingASR(
    std::shared_ptr<parakeet::IParakeetModel> model,
    const BufferedStreamingConfig& config
) : model_(model)
  , config_(config)
  , samples_received_(0)
  , samples_processed_(0)
{
    if (!model_) {
        throw std::invalid_argument("Model cannot be null");
    }

    if (config_.chunk_len_in_secs <= 0 || config_.total_buffer_in_secs <= 0) {
        throw std::invalid_argument("Chunk and buffer lengths must be positive");
    }

    if (config_.chunk_len_in_secs >= config_.total_buffer_in_secs) {
        throw std::invalid_argument("Chunk length must be less than total buffer size");
    }

    std::cout << "[STREAMING] Initialized with:\n";
    std::cout << "  Center chunk: " << config_.chunk_len_in_secs << "s ("
              << config_.center_chunk_samples() << " samples)\n";
    std::cout << "  Left context: " << (config_.left_context_samples() / (float)config_.sample_rate) << "s\n";
    std::cout << "  Right context: " << (config_.right_context_samples() / (float)config_.sample_rate) << "s\n";
    std::cout << "  Total buffer: " << config_.total_buffer_in_secs << "s\n";
}

StreamingResult BufferedStreamingASR::process_chunk(const std::vector<float>& audio_chunk) {
    // Add new audio to buffer
    audio_buffer_.insert(audio_buffer_.end(), audio_chunk.begin(), audio_chunk.end());
    samples_received_ += audio_chunk.size();

    // Check if we have enough data to process
    if (!has_enough_data()) {
        // Not enough data yet, wait for more
        return StreamingResult{
            .text = "",
            .token_ids = {},
            .confidence = 0.0f,
            .is_final = false
        };
    }

    // Process current window
    return process_current_window();
}

StreamingResult BufferedStreamingASR::finalize() {
    // Process any remaining audio in the buffer
    if (audio_buffer_.empty()) {
        return StreamingResult{
            .text = "",
            .token_ids = {},
            .confidence = 0.0f,
            .is_final = true
        };
    }

    // Pad buffer to required size if needed
    int required_size = config_.total_buffer_samples();
    if (audio_buffer_.size() < static_cast<size_t>(required_size)) {
        std::cout << "[STREAMING] Padding final buffer from " << audio_buffer_.size()
                  << " to " << required_size << " samples\n";
        audio_buffer_.resize(required_size, 0.0f);
    }

    // Process final window
    auto result = process_current_window();
    result.is_final = true;

    // Clear buffer
    audio_buffer_.clear();

    return result;
}

void BufferedStreamingASR::reset() {
    audio_buffer_.clear();
    previous_tokens_.clear();
    samples_received_ = 0;
    samples_processed_ = 0;
    std::cout << "[STREAMING] State reset\n";
}

StreamingResult BufferedStreamingASR::process_current_window() {
    // Extract the 10-second window from buffer
    int window_size = config_.total_buffer_samples();
    std::vector<float> window_audio(audio_buffer_.begin(),
                                    audio_buffer_.begin() + window_size);

    // Prepare audio segment
    parakeet::AudioSegment segment;
    segment.sample_rate = config_.sample_rate;
    segment.pcm = std::move(window_audio);

    // Run inference
    parakeet::SegmentOptions options;
    auto result = model_->infer(segment, options);

    // Remove overlapping tokens with previous window
    std::vector<int> non_overlap_tokens;
    if (!previous_tokens_.empty()) {
        non_overlap_tokens = remove_token_overlap(previous_tokens_, result.token_ids);
    } else {
        // First window, no overlap to remove
        non_overlap_tokens = result.token_ids;
    }

    // Update previous tokens for next iteration
    previous_tokens_ = result.token_ids;

    // Slide the window forward by center_chunk_samples
    int slide_amount = config_.center_chunk_samples();
    if (audio_buffer_.size() >= static_cast<size_t>(slide_amount)) {
        audio_buffer_.erase(audio_buffer_.begin(),
                           audio_buffer_.begin() + slide_amount);
    } else {
        audio_buffer_.clear();
    }
    samples_processed_ += slide_amount;

    std::cout << "[STREAMING] Processed window: "
              << non_overlap_tokens.size() << " tokens, "
              << "buffer: " << audio_buffer_.size() << " samples remaining\n";

    // Decode only the non-overlapping tokens to avoid duplicates
    std::string decoded_text = model_->decode_tokens(non_overlap_tokens);

    return StreamingResult{
        .text = decoded_text,
        .token_ids = non_overlap_tokens,
        .confidence = result.overall_confidence,
        .is_final = false
    };
}

std::vector<int> BufferedStreamingASR::remove_token_overlap(
    const std::vector<int>& prev_tokens,
    const std::vector<int>& current_tokens
) {
    // Algorithm: Find longest matching subsequence between end of prev and start of current
    // This is a 2D search as described in the document

    const int tail_size = std::min(20, static_cast<int>(prev_tokens.size()));
    const int head_size = std::min(15, static_cast<int>(current_tokens.size()));

    // Get tail of previous and head of current
    std::vector<int> prev_tail(
        prev_tokens.end() - tail_size,
        prev_tokens.end()
    );

    std::vector<int> curr_head(
        current_tokens.begin(),
        current_tokens.begin() + head_size
    );

    // Find longest match
    int best_match_len = 0;
    int best_match_start = 0;

    // Try all possible starting positions in prev_tail
    for (int i = 0; i < tail_size; ++i) {
        // Try all possible starting positions in curr_head
        for (int j = 0; j < head_size; ++j) {
            // Find match length starting from these positions
            int match_len = 0;
            while (i + match_len < tail_size &&
                   j + match_len < head_size &&
                   prev_tail[i + match_len] == curr_head[j + match_len]) {
                ++match_len;
            }

            // Update best match if this is longer
            if (match_len > best_match_len) {
                best_match_len = match_len;
                best_match_start = j + match_len; // Position after match in curr_head
            }
        }
    }

    if (best_match_len > 0) {
        std::cout << "[STREAMING] Found overlap of " << best_match_len
                  << " tokens, removing from current output\n";

        // Return tokens after the overlap
        // If overlap found in head, skip those tokens
        if (best_match_start > 0) {
            return std::vector<int>(
                current_tokens.begin() + best_match_start,
                current_tokens.end()
            );
        }
    }

    // No overlap found, return all current tokens
    std::cout << "[STREAMING] No overlap found, returning all " << current_tokens.size() << " tokens\n";
    return current_tokens;
}

} // namespace eddy::streaming
