#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void QwenTokenizer::build_byte_encoder() {
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint32_t codepoint = cs[i];
        std::string utf8_char;
        if (codepoint <= 0x7F) {
            utf8_char += static_cast<char>(codepoint);
        } else if (codepoint <= 0x7FF) {
            utf8_char += static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        byte_encoder[static_cast<uint8_t>(bs[i])] = utf8_char;
        byte_decoder[utf8_char] = static_cast<uint8_t>(bs[i]);
    }
}

QwenTokenizer::QwenTokenizer(const std::string& ranks_json_path) {
    std::ifstream f(ranks_json_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + ranks_json_path);
    }
    json j;
    f >> j;

    for (auto& [key, val] : j["vocab"].items()) {
        std::string token_str = key;
        int token_id = val.get<int>();
        encoder[token_str] = token_id;
        decoder[token_id] = token_str;
    }

    if (j.contains("added_tokens")) {
        for (auto& [key, val] : j["added_tokens"].items()) {
            int token_id = std::stoi(key);
            std::string token_str = val.get<std::string>();
            special_tokens[token_str] = token_id;
            sorted_special_tokens.push_back(token_str);
        }
    }

    std::sort(sorted_special_tokens.begin(), sorted_special_tokens.end(),
              [](const std::string& a, const std::string& b) {
                  return a.length() > b.length();
              });

    build_byte_encoder();

    // Qwen2.5 BPE Regex pattern
    std::string pattern = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    
    int errorcode;
    PCRE2_SIZE erroroffset;
    uint32_t options = PCRE2_UTF | PCRE2_UCP;

    re = pcre2_compile((PCRE2_SPTR)pattern.c_str(), PCRE2_ZERO_TERMINATED, options, &errorcode, &erroroffset, NULL);
    if (re == NULL) {
        throw std::runtime_error("PCRE2 compilation failed!");
    }
    match_data = pcre2_match_data_create_from_pattern(re, NULL);
}

QwenTokenizer::~QwenTokenizer() {
    pcre2_match_data_free(match_data);
    pcre2_code_free(re);
}

std::string QwenTokenizer::apply_chat_template(const std::vector<Message>& messages, bool add_generation_prompt) {
    std::string formatted = "";
    for (const auto& msg : messages) {
        formatted += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        formatted += "<|im_start|>assistant\n";
    }
    return formatted;
}

std::vector<int> QwenTokenizer::bpe(const std::string& chunk) {
    std::vector<std::string> parts;
    for (uint8_t b : chunk) {
        parts.push_back(byte_encoder[b]);
    }

    while (parts.size() >= 2) {
        int min_rank = std::numeric_limits<int>::max();
        int best_pair_idx = -1;
        std::string best_merged_str = "";

        for (size_t i = 0; i < parts.size() - 1; ++i) {
            std::string pair_str = parts[i] + parts[i + 1];
            auto it = encoder.find(pair_str);
            if (it != encoder.end()) {
                int rank = it->second;
                if (rank < min_rank) {
                    min_rank = rank;
                    best_pair_idx = static_cast<int>(i);
                    best_merged_str = pair_str;
                }
            }
        }

        if (best_pair_idx == -1) break;

        std::vector<std::string> new_parts;
        for (int i = 0; i < static_cast<int>(parts.size()); ++i) {
            if (i == best_pair_idx) {
                new_parts.push_back(best_merged_str);
                i++;
            } else {
                new_parts.push_back(parts[i]);
            }
        }
        parts = new_parts;
    }

    std::vector<int> tokens;
    for (const auto& part : parts) {
        tokens.push_back(encoder[part]);
    }
    return tokens;
}

std::vector<int> QwenTokenizer::encode_chunk(const std::string& text) {
    std::vector<int> tokens;
    PCRE2_SPTR subject = (PCRE2_SPTR)text.c_str();
    PCRE2_SIZE subject_length = text.length();
    PCRE2_SIZE start_offset = 0;

    while (start_offset < subject_length) {
        int rc = pcre2_match(re, subject, subject_length, start_offset, 0, match_data, NULL);
        if (rc < 0) break;

        PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
        PCRE2_SIZE start = ovector[0];
        PCRE2_SIZE end = ovector[1];

        std::string chunk = text.substr(start, end - start);
        std::vector<int> chunk_tokens = bpe(chunk);
        tokens.insert(tokens.end(), chunk_tokens.begin(), chunk_tokens.end());

        start_offset = end;
    }
    return tokens;
}

std::vector<int> QwenTokenizer::encode(const std::string& text) {
    std::vector<int> tokens;
    size_t pos = 0;

    while (pos < text.length()) {
        size_t next_special_pos = std::string::npos;
        std::string matched_special = "";
        int special_id = -1;

        for (const auto& st_str : sorted_special_tokens) {
            size_t found = text.find(st_str, pos);
            if (found != std::string::npos && found < next_special_pos) {
                next_special_pos = found;
                matched_special = st_str;
                special_id = special_tokens[st_str];
            }
        }

        if (next_special_pos == std::string::npos) {
            std::vector<int> chunk_toks = encode_chunk(text.substr(pos));
            tokens.insert(tokens.end(), chunk_toks.begin(), chunk_toks.end());
            break;
        } else {
            if (next_special_pos > pos) {
                std::vector<int> chunk_toks = encode_chunk(text.substr(pos, next_special_pos - pos));
                tokens.insert(tokens.end(), chunk_toks.begin(), chunk_toks.end());
            }
            tokens.push_back(special_id);
            pos = next_special_pos + matched_special.length();
        }
    }
    return tokens;
}

std::string QwenTokenizer::decode(const std::vector<int>& token_ids) {
    std::string raw_bytes = "";
    for (int id : token_ids) {
        raw_bytes += decode_single(id);
    }
    return raw_bytes;
}

std::string QwenTokenizer::decode_single(int token_id) {
    if (decoder.find(token_id) == decoder.end()) return "";
    std::string token_str = decoder[token_id];
    
    if (special_tokens.find(token_str) != special_tokens.end()) {
        return token_str;
    }

    std::string raw_bytes = "";
    size_t i = 0;
    while (i < token_str.size()) {
        bool matched = false;
        for (int len = 4; len >= 1; --len) {
            if (i + len <= token_str.size()) {
                std::string sub = token_str.substr(i, len);
                if (byte_decoder.find(sub) != byte_decoder.end()) {
                    raw_bytes += static_cast<char>(byte_decoder[sub]);
                    i += len;
                    matched = true;
                    break;
                }
            }
        }
        if (!matched) {
            raw_bytes += token_str[i];
            i++;
        }
    }
    return raw_bytes;
}