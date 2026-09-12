#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

#include "config.h"
#include "load_weights.h"
#include "model.cuh"
#include "tokenizer.h"

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << "         CUQWEN: Bare-Metal CUDA Chat Engine (Qwen2.5 " << QwenConfig::model_name << ")\n";
    std::cout << "========================================================================\n" << std::endl;

    std::string tokenizer_path = "weights/qwen25_bpe_ranks.json";
    std::cout << "[*] Loading Tokenizer from: " << tokenizer_path << "..." << std::flush;
    QwenTokenizer tokenizer(tokenizer_path);
    std::cout << " Done." << std::endl;

    const int eos_id = 151643;
    const int im_end_id = 151645;

    std::string bin_path = QwenConfig::bin_path;
    QwenWeights weights;
    QwenConfig config = load_weights(bin_path, weights);

    QwenState state;
    malloc_qwen_state(config, state);

    std::cout << "\n[✔] Engine Initialized and Ready!" << std::endl;
    std::cout << "Type 'exit' or 'quit' to terminate session." << std::endl;
    std::cout << "========================================================================\n" << std::endl;

    std::vector<Message> conversation_history;
    int current_pos = 0;

    // Reserve safety margin at the context boundary
    const int max_allowed_pos = QwenConfig::max_seq_len - 2;

    while (true) {
        if (current_pos >= max_allowed_pos) {
            std::cout << "\n[!] Context window fully utilized (" 
                      << current_pos << "/" << QwenConfig::max_seq_len 
                      << "). Terminating session gracefully..." << std::endl;
            break;
        }

        std::cout << "\nUser > ";
        std::string user_input;
        if (!std::getline(std::cin, user_input) || user_input == "exit" || user_input == "quit") {
            break;
        }

        if (user_input.empty()) continue;

        std::vector<int> prompt_tokens;

        if (conversation_history.empty()) {
            conversation_history.push_back({"user", user_input});
            std::string formatted_prompt = tokenizer.apply_chat_template(conversation_history, true);
            prompt_tokens = tokenizer.encode(formatted_prompt);
        } else {
            conversation_history.push_back({"user", user_input});
            std::vector<Message> new_turn = {{"user", user_input}};
            std::string formatted_new_turn = tokenizer.apply_chat_template(new_turn, true);
            prompt_tokens = tokenizer.encode(formatted_new_turn);
        }

        if (current_pos + static_cast<int>(prompt_tokens.size()) >= max_allowed_pos) {
            std::cout << "\n[!] Prompt too long for remaining context capacity (" 
                      << current_pos << "/" << QwenConfig::max_seq_len 
                      << "). Gracefully terminating session." << std::endl;
            break;
        }

        std::cout << "Qwen > " << std::flush;

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<int> generated_tokens;
        int num_generated = 0;

        // --- Prefill Phase ---
        int next_token = 0;
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            int tok = prompt_tokens[i];
            next_token = qwen_forward(tok, current_pos, config, state, weights);
            current_pos++;
        }

        // --- Decode Phase ---
        while (next_token != eos_id && next_token != im_end_id && current_pos < max_allowed_pos) {
            generated_tokens.push_back(next_token);
            num_generated++;

            std::string piece = tokenizer.decode_single(next_token);
            std::cout << piece << std::flush;

            next_token = qwen_forward(
                next_token,
                current_pos,
                config,
                state,
                weights,
                generated_tokens.data(),
                static_cast<int>(generated_tokens.size())
            );
            current_pos++;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(end_time - start_time).count();
        double tok_per_sec = (num_generated > 0) ? (num_generated / elapsed_sec) : 0.0;

        // Display performance stats
        std::cout << "\n\n[" << num_generated << " tokens generated | " 
                  << std::fixed << std::setprecision(2) << tok_per_sec << " tok/s | Context: "
                  << current_pos << "/" << QwenConfig::max_seq_len << "]" << std::endl;

        std::string assistant_response = tokenizer.decode(generated_tokens);
        conversation_history.push_back({"assistant", assistant_response});

        // Gracefully terminate post-generation if decode reached the threshold boundary
        if (current_pos >= max_allowed_pos) {
            std::cout << "\n[!] Maximum context window reached (" 
                      << current_pos << "/" << QwenConfig::max_seq_len 
                      << "). Session complete." << std::endl;
            break;
        }
    }

    std::cout << "\n[*] Cleaning up CUDA resources..." << std::flush;
    free_qwen_state(state);
    free_weights(weights, config);
    std::cout << " Done." << std::endl;

    return 0;
}