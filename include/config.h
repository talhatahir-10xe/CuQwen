#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// Weight quantization mode (selected at compile time via -Dquant=<fp16|int8|int4>).
// INT8/INT4 are weights-only (W8A16 / W4A16): the linear-projection weights are
// stored quantized + per-group FP16 scales and dequantized to FP16 inside the
// kernels. INT4 additionally packs 2 weights per byte (low/high nibble).
// Embeddings/LM head, RMSNorm weights and biases always remain FP16.
// ---------------------------------------------------------------------------
constexpr int QUANT_GROUP_SIZE = 128;   // weights sharing one scale (along input dim)

#if defined(QUANT_INT8)
using qweight_t = int8_t;               // one INT8 weight per element
#define QUANT_TAG "int8"
#define QUANT_TYPE_ID 1
#define QUANT_ENABLED
#elif defined(QUANT_INT4)
using qweight_t = int8_t;               // packed container: 2 x INT4 weights per byte
#define QUANT_TAG "int4"
#define QUANT_TYPE_ID 2
#define QUANT_ENABLED
#else
using qweight_t = half;                 // FP16 build: weights stored directly as half
#define QUANT_TAG "fp16"
#define QUANT_TYPE_ID 0
#endif

// Stride (in units of qweight_t) of one weight row holding `cols` logical
// weights. INT4 packs 2 weights per byte, so a row occupies cols/2 bytes.
#ifdef QUANT_INT4
#define WEIGHT_ROW_STRIDE(cols) ((cols) / 2)
#else
#define WEIGHT_ROW_STRIDE(cols) (cols)
#endif

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
    #define MODEL_TAG "0_5b"
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
    #define MODEL_TAG "1_5b"
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
    #define MODEL_TAG "3b"
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
    #define MODEL_TAG "7b"
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

    // Binary path and quant tag are composed from the model + quantization mode,
    // e.g. "weights/model_int8_3b.bin". (Adjacent string literals concatenate.)
    static constexpr const char* bin_path   = "weights/model_" QUANT_TAG "_" MODEL_TAG ".bin";
    static constexpr int32_t quant_type     = QUANT_TYPE_ID;

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
