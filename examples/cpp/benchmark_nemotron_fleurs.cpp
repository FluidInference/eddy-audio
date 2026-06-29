// Copyright (C) 2025 Eddy SDK
// SPDX-License-Identifier: Apache-2.0
//
// FLEURS Multilingual ASR Benchmark for the Nemotron streaming backend.
//
// Mirrors benchmark_fleurs.cpp (Parakeet) but drives eddy::nemotron::
// OpenVINONemotron with per-language prompt conditioning. Loads the model once
// per language (language is fixed at handle construction), loops the FLEURS
// split, and reports WER / CER / RTFx.
//
// Usage:
//   benchmark_nemotron_fleurs.exe <fleurs_cache_dir> --languages en_us,es_419,fr_fr \
//       --samples 0 --device NPU --model-dir <dir with nemotron_*.xml> --output results.json

#include "eddy/backends/openvino_backend.hpp"
#include "eddy/core/app_dir.hpp"
#include "eddy/models/nemotron/nemotron.hpp"
#include "eddy/utils/audio_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// FLEURS code -> human name (subset; only used for display).
const std::map<std::string, std::string> LANG_NAMES = {
    {"en_us", "English (US)"},     {"es_419", "Spanish (LatAm)"}, {"fr_fr", "French (France)"},
    {"de_de", "German (Germany)"}, {"it_it", "Italian (Italy)"},  {"ru_ru", "Russian (Russia)"},
    {"nl_nl", "Dutch"},            {"pl_pl", "Polish"},           {"uk_ua", "Ukrainian"},
    {"sk_sk", "Slovak"},           {"cs_cz", "Czech"},            {"bg_bg", "Bulgarian"},
    {"hr_hr", "Croatian"},         {"ro_ro", "Romanian"},         {"fi_fi", "Finnish"},
    {"hu_hu", "Hungarian"},        {"sv_se", "Swedish"},          {"et_ee", "Estonian"},
    {"da_dk", "Danish"},           {"lt_lt", "Lithuanian"},       {"el_gr", "Greek"},
    {"mt_mt", "Maltese"},          {"lv_lv", "Latvian"},          {"sl_si", "Slovenian"},
    {"cmn_hans_cn", "Chinese (Mandarin)"}, {"ja_jp", "Japanese"},
};

// Map a FLEURS code ("en_us") to a Nemotron prompt-dictionary tag ("en-US").
// es_419 (Latin-American Spanish) is special-cased to es-US; everything else is
// the generic "xx_yy" -> "xx-YY". Unknown tags fall back inside the backend
// (2-letter, then "auto").
std::string fleurs_to_nemotron_lang(const std::string& code) {
    if (code == "es_419") return "es-US";
    if (code == "cmn_hans_cn") return "zh-CN";  // FLEURS Mandarin -> Nemotron zh-CN
    if (code == "ja_jp") return "ja-JP";
    auto us = code.find('_');
    if (us == std::string::npos) return code;
    std::string lang = code.substr(0, us);
    std::string region = code.substr(us + 1);
    for (char& c : region) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return lang + "-" + region;
}

// Split a UTF-8 string into codepoint substrings (1-4 bytes each).
std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> cps;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = (c < 0x80) ? 1 : ((c >> 5) == 0x6) ? 2 : ((c >> 4) == 0xE) ? 3
                     : ((c >> 3) == 0x1E) ? 4 : 1;
        if (i + len > s.size()) len = 1;
        cps.push_back(s.substr(i, len));
        i += len;
    }
    return cps;
}

// Decode a 1-4 byte UTF-8 codepoint string to its Unicode scalar value.
uint32_t cp_scalar(const std::string& cp) {
    unsigned char c0 = static_cast<unsigned char>(cp[0]);
    if (cp.size() == 1) return c0;
    if (cp.size() == 2) return ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(cp[1]) & 0x3F);
    if (cp.size() == 3)
        return ((c0 & 0x0F) << 12) | ((static_cast<unsigned char>(cp[1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(cp[2]) & 0x3F);
    return ((c0 & 0x07) << 18) | ((static_cast<unsigned char>(cp[1]) & 0x3F) << 12) |
           ((static_cast<unsigned char>(cp[2]) & 0x3F) << 6) | (static_cast<unsigned char>(cp[3]) & 0x3F);
}

// Approximate the Whisper/FluidAudio normalizer's "replace every Mark/Symbol/
// Punctuation (Unicode category M/S/P) with a space" step over the codepoint
// blocks that actually occur in FLEURS refs/hyps. Keeps letters (incl. CJK,
// kana, hangul, accented Latin, Greek, Cyrillic) and digits.
bool is_punct_or_symbol(uint32_t c) {
    return (c >= 0x00A1 && c <= 0x00BF) || c == 0x00D7 || c == 0x00F7 ||  // Latin-1 punct/symbols, × ÷
           (c >= 0x2000 && c <= 0x206F) ||   // general punctuation – — ' ' " " …
           (c >= 0x2070 && c <= 0x20CF) ||   // super/subscripts, currency symbols
           (c >= 0x2100 && c <= 0x2BFF) ||   // letterlike/number forms, arrows, math, misc symbols
           (c >= 0x3000 && c <= 0x303F) ||   // CJK symbols and punctuation 。、「」（）
           (c >= 0xFF01 && c <= 0xFF0F) ||   // fullwidth ！＂＃…／
           (c >= 0xFF1A && c <= 0xFF20) ||   // fullwidth ：；＜＝＞？＠
           (c >= 0xFF3B && c <= 0xFF40) ||   // fullwidth ［＼］＾＿｀
           (c >= 0xFF5B && c <= 0xFF65);     // fullwidth ｛｜｝、。etc
}

// Normalize ~ FluidAudio's basicNormalize: lowercase ASCII, replace Unicode
// punctuation/symbols (M/S/P) with single spaces, keep letters/digits and
// diacritics/CJK, collapse whitespace. (Whisper's English number-word folding
// is intentionally omitted — "similar enough" per the multilingual path.)
std::string normalize_text(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    bool last_space = false;
    auto sep = [&]() { if (!last_space && !result.empty()) { result += ' '; last_space = true; } };
    for (const std::string& cp : utf8_chars(text)) {
        if (cp.size() == 1) {
            unsigned char c = static_cast<unsigned char>(cp[0]);
            if (std::isalnum(c)) { result += static_cast<char>(std::tolower(c)); last_space = false; }
            else sep();
        } else if (is_punct_or_symbol(cp_scalar(cp))) {
            sep();
        } else {
            result += cp;  // letter / CJK / diacritic — keep
            last_space = false;
        }
    }
    if (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

// CJK / no-space scripts: word-level WER over whitespace tokens is meaningless,
// so FluidAudio routes these through character-level scoring (matches Whisper /
// ESPnet). FLEURS code prefixes.
bool is_cjk_lang(const std::string& code) {
    auto p = [&](const char* s) { return code.rfind(s, 0) == 0; };
    return p("ja") || p("ko") || p("zh") || p("cmn") || p("yue") || p("th") || p("lo");
}

int levenshtein(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    const size_t m = a.size(), n = b.size();
    std::vector<int> prev(n + 1), cur(n + 1);
    for (size_t j = 0; j <= n; ++j) prev[j] = static_cast<int>(j);
    for (size_t i = 1; i <= m; ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= n; ++j) {
            if (a[i - 1] == b[j - 1]) cur[j] = prev[j - 1];
            else cur[j] = 1 + std::min({prev[j], cur[j - 1], prev[j - 1]});
        }
        std::swap(prev, cur);
    }
    return prev[n];
}

std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> w;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) w.push_back(t);
    return w;
}

double wer(const std::string& ref, const std::string& hyp) {
    auto r = words(normalize_text(ref)), h = words(normalize_text(hyp));
    if (r.empty()) return h.empty() ? 0.0 : 1.0;
    return static_cast<double>(levenshtein(r, h)) / r.size();
}

double cer(const std::string& ref, const std::string& hyp) {
    // Character error rate over UTF-8 codepoints (spaces dropped). Codepoint-
    // level, so CJK characters count as one token each.
    std::vector<std::string> r, h;
    for (const auto& cp : utf8_chars(normalize_text(ref))) if (cp != " ") r.push_back(cp);
    for (const auto& cp : utf8_chars(normalize_text(hyp))) if (cp != " ") h.push_back(cp);
    if (r.empty()) return h.empty() ? 0.0 : 1.0;
    return static_cast<double>(levenshtein(r, h)) / r.size();
}

struct Sample { std::string id, audio_path, transcription; };

std::vector<Sample> load_samples(const std::string& cache_dir, const std::string& lang, int max_samples) {
    std::vector<Sample> samples;
    fs::path dir = fs::path(cache_dir) / lang;
    if (!fs::exists(dir)) {
        std::cerr << "Warning: language dir not found: " << dir << "\n";
        return samples;
    }
    std::map<std::string, std::string> trans;
    fs::path tf = dir / (lang + ".trans.txt");
    if (fs::exists(tf)) {
        std::ifstream f(tf);
        std::string line;
        while (std::getline(f, line)) {
            size_t sp = line.find(' ');
            if (sp != std::string::npos) trans[line.substr(0, sp)] = line.substr(sp + 1);
        }
    }
    std::vector<fs::path> wavs;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().extension() == ".wav") wavs.push_back(e.path());
    std::sort(wavs.begin(), wavs.end());
    if (max_samples > 0 && wavs.size() > static_cast<size_t>(max_samples))
        wavs.resize(max_samples);
    for (const auto& w : wavs) {
        std::string id = w.stem().string();
        Sample s{id, w.string(), trans.count(id) ? trans[id] : ""};
        samples.push_back(s);
    }
    return samples;
}

struct LangResult {
    std::string lang, name;
    double wer = 0, cer = 0, rtfx = 0, total_audio = 0, total_proc = 0;
    int processed = 0, skipped = 0;
};

void print_usage(const char* p) {
    std::cout << "Nemotron FLEURS Benchmark\n\nUsage: " << p << " <cache_dir> [options]\n"
              << "  --languages <list>  Comma-separated FLEURS codes (default: en_us,es_419,fr_fr)\n"
              << "  --samples <n>       Max samples per language (0 = all; default 0)\n"
              << "  --device <dev>      OpenVINO device CPU/NPU/GPU/AUTO (default CPU)\n"
              << "  --model-dir <dir>   Dir with nemotron_*.xml/bin + metadata.json\n"
              << "  --output <file>     Output JSON (default nemotron_fleurs_results.json)\n"
              << "  --debug             Per-file hypothesis/reference\n";
}

int main(int argc, char* argv[]) {
    std::cout.setf(std::ios::unitbuf);
    if (argc < 2) { print_usage(argv[0]); return 1; }

    std::string cache_dir = argv[1];
    std::vector<std::string> langs = {"en_us", "es_419", "fr_fr"};
    int max_samples = 0;
    std::string device = "CPU", model_dir, output = "nemotron_fleurs_results.json";
    bool debug = false;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string& dst) { if (i + 1 < argc) dst = argv[++i]; };
        if (a == "--help" || a == "-h") { print_usage(argv[0]); return 0; }
        else if (a == "--languages") { std::string v; next(v); langs.clear();
            std::istringstream ss(v); std::string t; while (std::getline(ss, t, ',')) langs.push_back(t); }
        else if (a == "--samples") { std::string v; next(v); max_samples = std::stoi(v); }
        else if (a == "--device") next(device);
        else if (a == "--model-dir") next(model_dir);
        else if (a == "--output") next(output);
        else if (a == "--debug") debug = true;
    }

    // Default model dir: %LOCALAPPDATA%/eddy/models/nemotron-streaming-int8/files
    if (model_dir.empty()) {
        const char* lad = std::getenv("LOCALAPPDATA");
        if (lad) model_dir = (fs::path(lad) / "eddy" / "models" / "nemotron-streaming-int8" / "files").string();
    }

    std::cout << "=== Nemotron FLEURS Benchmark ===\n";
    std::cout << "Cache:      " << cache_dir << "\n";
    std::cout << "Model dir:  " << model_dir << "\n";
    std::cout << "Device:     " << device << "\n";
    std::cout << "Samples:    " << (max_samples == 0 ? "all" : std::to_string(max_samples)) << " per language\n\n";

    eddy::nemotron::ModelPaths paths{
        .preprocessor  = (fs::path(model_dir) / "nemotron_preprocessor.xml").string(),
        .encoder       = (fs::path(model_dir) / "nemotron_encoder.xml").string(),
        .decoder       = (fs::path(model_dir) / "nemotron_decoder.xml").string(),
        .joint         = (fs::path(model_dir) / "nemotron_joint.xml").string(),
        .vocab_json    = (fs::path(model_dir) / "nemotron_vocab.json").string(),
        .metadata_json = (fs::path(model_dir) / "metadata.json").string(),
    };

    eddy::OpenVINOOptions ov_opts;
    ov_opts.device = device;
    ov_opts.cache_dir = eddy::get_model_dir("nemotron-streaming-int8").string();
    auto backend = std::make_shared<eddy::OpenVINOBackend>(ov_opts);

    std::vector<LangResult> results;
    for (const auto& lang : langs) {
        std::string tag = fleurs_to_nemotron_lang(lang);
        std::cout << "Processing " << lang << " (prompt lang=" << tag << ")...\n";

        auto samples = load_samples(cache_dir, lang, max_samples);
        if (samples.empty()) { std::cerr << "  no samples; skipping\n\n"; continue; }
        std::cout << "  Loaded " << samples.size() << " samples\n";

        eddy::nemotron::Config cfg;
        cfg.device = device;
        cfg.language = tag;

        std::shared_ptr<eddy::nemotron::OpenVINONemotron> model;
        try {
            model = std::make_shared<eddy::nemotron::OpenVINONemotron>(backend, paths, cfg);
            std::cout << "  Compiling + warming up (" << device << ") ... ";
            model->warmup();
            std::cout << "[OK]\n";
        } catch (const std::exception& e) {
            std::cerr << "  [ERROR] model init failed: " << e.what() << "\n\n";
            continue;
        }

        LangResult lr;
        lr.lang = lang;
        lr.name = LANG_NAMES.count(lang) ? LANG_NAMES.at(lang) : lang;
        const bool cjk = is_cjk_lang(lang);
        double sum_wer = 0, sum_cer = 0;

        for (const auto& s : samples) {
            try {
                auto pcm = eddy::audio::read_wav(s.audio_path);
                double audio_sec = pcm.size() / 16000.0;
                auto t0 = std::chrono::high_resolution_clock::now();
                auto res = model->transcribe(pcm);
                auto t1 = std::chrono::high_resolution_clock::now();
                double proc_sec = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

                if (!s.transcription.empty()) {
                    // CJK: character-level rate reported in both WER and CER
                    // (FluidAudio convention — whitespace WER is meaningless).
                    double c = cer(s.transcription, res.text);
                    double w = cjk ? c : wer(s.transcription, res.text);
                    sum_wer += w; sum_cer += c;
                    if (debug) {
                        std::cout << "  [" << s.id << "] WER=" << std::fixed << std::setprecision(1) << (w * 100) << "%\n"
                                  << "    hyp: " << res.text << "\n    ref: " << s.transcription << "\n";
                    }
                }
                lr.total_audio += audio_sec;
                lr.total_proc += proc_sec;
                lr.processed++;
            } catch (const std::exception& e) {
                std::cerr << "  Warning: " << s.id << ": " << e.what() << "\n";
                lr.skipped++;
            }
        }

        if (lr.processed > 0) {
            lr.wer = sum_wer / lr.processed;
            lr.cer = sum_cer / lr.processed;
            lr.rtfx = lr.total_proc > 0 ? lr.total_audio / lr.total_proc : 0;
        }
        results.push_back(lr);
        std::cout << "  " << lang << ": WER=" << std::fixed << std::setprecision(2) << (lr.wer * 100)
                  << "%  CER=" << (lr.cer * 100) << "%  RTFx=" << std::setprecision(2) << lr.rtfx
                  << "x  (" << lr.processed << " processed, " << lr.skipped << " skipped)\n\n";
    }

    // JSON output
    {
        std::ofstream f(output);
        f << "{\n  \"benchmark\": \"FLEURS Nemotron streaming\",\n  \"device\": \"" << device
          << "\",\n  \"model_dir\": \"";
        for (char c : model_dir) { if (c == '\\') f << "\\\\"; else f << c; }
        f << "\",\n  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            f << "    {\"language\": \"" << r.lang << "\", \"wer\": " << (r.wer * 100)
              << ", \"cer\": " << (r.cer * 100) << ", \"rtfx\": " << r.rtfx
              << ", \"processed\": " << r.processed << ", \"skipped\": " << r.skipped
              << ", \"audioSec\": " << r.total_audio << ", \"procSec\": " << r.total_proc << "}";
            f << (i + 1 < results.size() ? ",\n" : "\n");
        }
        f << "  ]\n}\n";
    }

    std::cout << std::string(72, '=') << "\nSUMMARY (device=" << device << ")\n" << std::string(72, '=') << "\n";
    double tot_audio = 0, tot_proc = 0;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(18) << r.name << " | WER=" << std::right << std::fixed
                  << std::setprecision(2) << std::setw(6) << (r.wer * 100) << "%  CER=" << std::setw(6)
                  << (r.cer * 100) << "%  RTFx=" << std::setw(5) << std::setprecision(2) << r.rtfx
                  << "x  n=" << r.processed << "\n";
        tot_audio += r.total_audio; tot_proc += r.total_proc;
    }
    std::cout << std::string(72, '-') << "\nAudio-weighted RTFx: " << std::fixed << std::setprecision(2)
              << (tot_proc > 0 ? tot_audio / tot_proc : 0) << "x   (total audio "
              << std::setprecision(0) << tot_audio << "s / proc " << tot_proc << "s)\n";
    std::cout << "Results saved to: " << output << "\n";
    return 0;
}
