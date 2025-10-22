// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <unordered_map>
#include <regex>
#include <vector>
#include <sstream>
#include <cctype>
#include <fstream>
#include <algorithm>
#include <limits>
#include <iostream>

#include "nlohmann/json.hpp"

namespace eddy {

class TextNormalizer {
public:
    TextNormalizer() {
        initialize_mappings();
    }

    // Main normalization function - matches FluidAudio's approach
    std::string normalize(const std::string& text) const {
        std::string result = text;

        // 1. Lowercase
        result = to_lowercase(result);

        // 2. Expand numbers (including ranges like 23-33)
        result = expand_numbers(result);

        // 3. Expand abbreviations
        result = expand_abbreviations(result);

        // 4. Expand contractions
        result = expand_contractions(result);

        // 5. Apply British→American variants (if dictionary is loaded)
        result = apply_variants(result);

        // 6. Remove punctuation except apostrophes in words
        result = clean_punctuation(result);

        // 7. Normalize whitespace
        result = normalize_whitespace(result);

        return result;
    }

    // Load a dictionary mapping (e.g., FluidAudio's english.json).
    // JSON must be an object of { "british": "american", ... }.
    void load_variants_json(const std::string& json_path) {
        try {
            std::ifstream in(json_path);
            if (!in.good()) return;
            nlohmann::json j; in >> j;
            if (!j.is_object()) return;

            variants_.clear();
            for (auto it = j.begin(); it != j.end(); ++it) {
                if (!it.value().is_string()) continue;
                std::string key = to_lowercase(it.key());
                std::string val = to_lowercase(it.value().get<std::string>());
                sanitize_token(key);
                sanitize_token(val);
                if (!key.empty() && !val.empty()) {
                    variants_[key] = val;
                }
            }
        } catch (const std::exception& e) {
            if (std::getenv("EDDY_DEBUG")) {
                std::cerr << "[DEBUG] Failed to load variants dictionary: " << e.what() << "\n";
            }
        } catch (...) {
            // Ignore unknown errors; normalization still works without variants
        }
    }

private:
    std::unordered_map<std::string, std::string> abbreviations_;
    std::unordered_map<std::string, std::string> contractions_;
    std::unordered_map<std::string, std::regex> contraction_patterns_;
    std::unordered_map<std::string, std::string> variants_;

    void initialize_mappings() {
        // Common abbreviations (matching FluidAudio)
        abbreviations_ = {
            {"mr", "mister"},
            {"mrs", "missus"},
            {"ms", "miss"},
            {"dr", "doctor"},
            {"prof", "professor"},
            {"rev", "reverend"},
            {"gen", "general"},
            {"col", "colonel"},
            {"lt", "lieutenant"},
            {"sr", "senior"},
            {"jr", "junior"},
            {"st", "saint"},
            {"ave", "avenue"},
            {"blvd", "boulevard"},
            {"rd", "road"},
            {"ln", "lane"},
            {"ct", "court"},
            {"etc", "etcetera"},
            {"vs", "versus"},
            {"inc", "incorporated"},
            {"ltd", "limited"},
            {"co", "company"},
            {"corp", "corporation"}
        };

        // Common contractions (matching FluidAudio)
        contractions_ = {
            {"don't", "do not"},
            {"won't", "will not"},
            {"can't", "cannot"},
            {"couldn't", "could not"},
            {"wouldn't", "would not"},
            {"shouldn't", "should not"},
            {"isn't", "is not"},
            {"aren't", "are not"},
            {"wasn't", "was not"},
            {"weren't", "were not"},
            {"hasn't", "has not"},
            {"haven't", "have not"},
            {"hadn't", "had not"},
            {"doesn't", "does not"},
            {"didn't", "did not"},
            {"i'm", "i am"},
            {"you're", "you are"},
            {"he's", "he is"},
            {"she's", "she is"},
            {"it's", "it is"},
            {"we're", "we are"},
            {"they're", "they are"},
            {"i've", "i have"},
            {"you've", "you have"},
            {"we've", "we have"},
            {"they've", "they have"},
            {"i'd", "i would"},
            {"you'd", "you would"},
            {"he'd", "he would"},
            {"she'd", "she would"},
            {"we'd", "we would"},
            {"they'd", "they would"},
            {"i'll", "i will"},
            {"you'll", "you will"},
            {"he'll", "he will"},
            {"she'll", "she will"},
            {"we'll", "we will"},
            {"they'll", "they will"},
            {"let's", "let us"},
            {"that's", "that is"},
            {"there's", "there is"},
            {"here's", "here is"},
            {"what's", "what is"},
            {"who's", "who is"},
            {"where's", "where is"},
            {"when's", "when is"},
            {"why's", "why is"},
            {"how's", "how is"}
        };

        // Pre-compile regex patterns for contractions (performance optimization)
        for (const auto& [contraction, expansion] : contractions_) {
            contraction_patterns_[contraction] =
                std::regex("\\b" + contraction + "\\b", std::regex_constants::icase);
        }
    }

    std::string to_lowercase(const std::string& text) const {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return result;
    }

    // Number to words conversion
    std::string number_to_words(int num) const {
        if (num == 0) return "zero";

        const std::vector<std::string> ones = {
            "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
            "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
            "seventeen", "eighteen", "nineteen"
        };

        const std::vector<std::string> tens = {
            "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
        };

        const std::vector<std::string> thousands = {
            "", "thousand", "million", "billion"
        };

        if (num < 0) {
            // Handle INT_MIN overflow case
            if (num == std::numeric_limits<int>::min()) {
                return "negative two billion one hundred forty seven million four hundred eighty three thousand six hundred forty eight";
            }
            return "negative " + number_to_words(-num);
        }

        if (num < 20) {
            return ones[num];
        }

        if (num < 100) {
            return tens[num / 10] + (num % 10 > 0 ? " " + ones[num % 10] : "");
        }

        if (num < 1000) {
            return ones[num / 100] + " hundred" +
                   (num % 100 > 0 ? " " + number_to_words(num % 100) : "");
        }

        // For larger numbers
        std::string result;
        int thousand_power = 0;

        while (num > 0) {
            int group = num % 1000;
            if (group != 0) {
                std::string group_str = number_to_words(group);
                if (thousand_power > 0 && thousand_power < thousands.size()) {
                    group_str += " " + thousands[thousand_power];
                }
                if (!result.empty()) {
                    result = group_str + " " + result;
                } else {
                    result = group_str;
                }
            }
            num /= 1000;
            thousand_power++;
        }

        return result;
    }

    std::string expand_numbers(const std::string& text) const {
        static const std::regex number_pattern(R"(\b(\d+)(?:\s*[-–]\s*(\d+))?\b)");
        std::string result = text;
        std::smatch match;

        while (std::regex_search(result, match, number_pattern)) {
            std::string replacement;
            int num1 = std::stoi(match[1].str());

            if (match[2].matched) {
                // Handle range like "23-33" -> "twenty three to thirty three"
                int num2 = std::stoi(match[2].str());
                replacement = number_to_words(num1) + " to " + number_to_words(num2);
            } else if (match[1].str().length() == 4 && num1 >= 1900 && num1 <= 2099) {
                // Handle years in common range
                if (num1 >= 2000) {
                    // 2023 -> "two thousand twenty three"
                    replacement = (num1 % 100 > 0)
                        ? "two thousand " + number_to_words(num1 % 100)
                        : "two thousand";
                } else {
                    // 1984 -> "nineteen eighty four"
                    int century = num1 / 100;
                    int year = num1 % 100;
                    replacement = number_to_words(century) + " " +
                                 (year < 10 ? "oh " + number_to_words(year) : number_to_words(year));
                }
            } else {
                replacement = number_to_words(num1);
            }

            result = result.substr(0, match.position()) + replacement +
                     result.substr(match.position() + match.length());
        }

        return result;
    }

    std::string expand_abbreviations(const std::string& text) const {
        static const std::regex word_pattern(R"(\b[a-z]+\.?\b)");
        std::string result;
        std::sregex_iterator it(text.begin(), text.end(), word_pattern);
        std::sregex_iterator end;

        size_t last_pos = 0;
        for (; it != end; ++it) {
            std::smatch match = *it;
            result += text.substr(last_pos, match.position() - last_pos);

            std::string word = match.str();
            // Remove trailing period if present
            if (!word.empty() && word.back() == '.') {
                word = word.substr(0, word.length() - 1);
            }

            // Check if it's an abbreviation
            auto abbr_it = abbreviations_.find(word);
            if (abbr_it != abbreviations_.end()) {
                result += abbr_it->second;
            } else {
                result += match.str();
            }

            last_pos = match.position() + match.length();
        }
        result += text.substr(last_pos);

        return result;
    }

    std::string expand_contractions(const std::string& text) const {
        std::string result = text;

        for (const auto& [contraction, expansion] : contractions_) {
            result = std::regex_replace(result, contraction_patterns_.at(contraction), expansion);
        }

        return result;
    }

    // Apply British→American word variants on word boundaries only
    std::string apply_variants(const std::string& text) const {
        if (variants_.empty()) return text;

        std::string result;
        result.reserve(text.size());

        // Tokenize keeping non-letters intact, operate on [a-z]+ tokens
        size_t i = 0, n = text.size();
        while (i < n) {
            if (std::isalpha(static_cast<unsigned char>(text[i]))) {
                size_t start = i;
                while (i < n && std::isalpha(static_cast<unsigned char>(text[i]))) i++;
                std::string word = text.substr(start, i - start);
                std::string lower = to_lowercase(word);
                auto it = variants_.find(lower);
                if (it != variants_.end()) {
                    result += it->second;
                } else {
                    result += word;
                }
            } else {
                result += text[i++];
            }
        }
        return result;
    }

    std::string clean_punctuation(const std::string& text) const {
        std::string result;
        bool in_word = false;

        for (size_t i = 0; i < text.length(); ++i) {
            char c = text[i];

            if (std::isalnum(c)) {
                result += c;
                in_word = true;
            } else if (c == '\'' && in_word && i + 1 < text.length() && std::isalpha(text[i + 1])) {
                // Keep apostrophes within words
                result += c;
            } else if (std::isspace(c)) {
                result += ' ';
                in_word = false;
            } else {
                // Skip other punctuation
                in_word = false;
            }
        }

        return result;
    }

    std::string normalize_whitespace(const std::string& text) const {
        std::string result;
        bool prev_space = true; // Start as true to trim leading spaces

        for (char c : text) {
            if (std::isspace(c)) {
                if (!prev_space) {
                    result += ' ';
                    prev_space = true;
                }
            } else {
                result += c;
                prev_space = false;
            }
        }

        // Trim trailing space
        if (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }

        return result;
    }

    // Sanitize tokens loaded from JSON (strip trivial HTML fragments, punctuation around word)
    static void sanitize_token(std::string& s) {
        // Remove any simple HTML tags like </span>
        static const std::regex html_pattern(R"(<[^>]*>)");
        s = std::regex_replace(s, html_pattern, "");
        // Trim surrounding whitespace
        auto not_space = [](int ch){ return !std::isspace(ch); };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
        s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
        // Keep only letters/apostrophes/hyphens in tokens we map
        std::string cleaned;
        cleaned.reserve(s.size());
        for (char c : s) {
            if (std::isalpha(static_cast<unsigned char>(c)) || c=='\'' || c=='-') cleaned.push_back(c);
        }
        s.swap(cleaned);
    }
};

} // namespace eddy
