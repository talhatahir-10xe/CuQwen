#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct QwenConfig {
    static constexpr int32_t magic = 0x5157454E;

    // Qwen2.5 1.5B Hyperparameters
    static constexpr int32_t vocab_size        = 151936;
    static constexpr int32_t dim               = 1536;
    static constexpr int32_t intermediate_size = 8960;
    static constexpr int32_t n_layers          = 28;
    static constexpr int32_t n_heads           = 12;
    static constexpr int32_t n_kv_heads        = 2;
    static constexpr int32_t head_dim          = 128;
    static constexpr int32_t max_seq_len       = 8192; // Adjust maximum sequence context length

    // Operational Constants
    static constexpr float norm_eps           = 1e-6f;
    static constexpr float rope_theta          = 1000000.0f;
    static constexpr float repetition_penalty = 1.1f;
};

#endif // CONFIG_H