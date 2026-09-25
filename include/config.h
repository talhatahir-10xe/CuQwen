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

#if defined(QWEN_MODEL_0_5B)
    static constexpr const char* model_name = "0.5B";
    static constexpr const char* bin_path   = "weights/model_fp16_0_5b.bin";
    static constexpr int32_t vocab_size        = 151936;
    static constexpr int32_t dim               = 896;
    static constexpr int32_t intermediate_size = 4864;
    static constexpr int32_t n_layers          = 24;
    static constexpr int32_t n_heads           = 14;
    static constexpr int32_t n_kv_heads        = 2;
    static constexpr int32_t head_dim          = 64;
    static constexpr int32_t max_seq_len       = 32768;

#elif defined(QWEN_MODEL_1_5B)
    static constexpr const char* model_name = "1.5B";
    static constexpr const char* bin_path   = "weights/model_fp16_1_5b.bin";
    static constexpr int32_t vocab_size        = 151936;
    static constexpr int32_t dim               = 1536;
    static constexpr int32_t intermediate_size = 8960;
    static constexpr int32_t n_layers          = 28;
    static constexpr int32_t n_heads           = 12;
    static constexpr int32_t n_kv_heads        = 2;
    static constexpr int32_t head_dim          = 128;
    static constexpr int32_t max_seq_len       = 32768;

#elif defined(QWEN_MODEL_3B)
    static constexpr const char* model_name = "3B";
    static constexpr const char* bin_path   = "weights/model_fp16_3b.bin";
    static constexpr int32_t vocab_size        = 151936;
    static constexpr int32_t dim               = 2048;
    static constexpr int32_t intermediate_size = 11008;
    static constexpr int32_t n_layers          = 36;
    static constexpr int32_t n_heads           = 16;
    static constexpr int32_t n_kv_heads        = 2;
    static constexpr int32_t head_dim          = 128;
    static constexpr int32_t max_seq_len       = 32768;

#elif defined(QWEN_MODEL_7B)
    static constexpr const char* model_name = "7B";
    static constexpr const char* bin_path   = "weights/model_fp16_7b.bin";
    static constexpr int32_t vocab_size        = 152064;
    static constexpr int32_t dim               = 3584;
    static constexpr int32_t intermediate_size = 18944;
    static constexpr int32_t n_layers          = 28;
    static constexpr int32_t n_heads           = 28;
    static constexpr int32_t n_kv_heads        = 4;
    static constexpr int32_t head_dim          = 128;
    static constexpr int32_t max_seq_len       = 32768;

#else
    #error "Model size preprocessor directive not set! Define QWEN_MODEL_0_5B, QWEN_MODEL_1_5B, QWEN_MODEL_3B, or QWEN_MODEL_7B."
#endif

    static constexpr float norm_eps           = 1e-6f;
    static constexpr float rope_theta          = 1000000.0f;
    static constexpr float repetition_penalty = 1.1f;
};

// Tuned for RTX 3090
constexpr int ATTN_PARTITIONS = 64;
constexpr int KSPLIT_WARPS    = 8;

constexpr int ATTN_KEYS_CAP =
    (((QwenConfig::max_seq_len + ATTN_PARTITIONS - 1) / ATTN_PARTITIONS) + 15) / 16 * 16;

constexpr int GEMV_WARPS_PER_BLOCK = 8;
constexpr int HEADS_PER_WARP = 4;

#endif // CONFIG_H
