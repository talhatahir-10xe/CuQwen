#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

struct Message {
    std::string role;
    std::string content;
};

class QwenTokenizer {
private:
    std::unordered_map<std::string, int> encoder;
    std::unordered_map<int, std::string> decoder;
    std::unordered_map<std::string, int> special_tokens;
    std::vector<std::string> sorted_special_tokens;
    
    std::unordered_map<uint8_t, std::string> byte_encoder;
    std::unordered_map<std::string, uint8_t> byte_decoder;

    pcre2_code* re;
    pcre2_match_data* match_data;

    void build_byte_encoder();
    std::vector<int> bpe(const std::string& chunk);
    std::vector<int> encode_chunk(const std::string& text);

public:
    QwenTokenizer(const std::string& ranks_json_path);
    ~QwenTokenizer();

    std::string apply_chat_template(const std::vector<Message>& messages, bool add_generation_prompt = true);
    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& token_ids);
    std::string decode_single(int token_id);
};

#endif // TOKENIZER_H