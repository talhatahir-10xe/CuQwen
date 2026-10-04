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

// ===========================================================================
// Performance tuning knobs (compile-time). Defaults below are tuned for the
// RTX 3090 (Ampere, 82 SMs, sm_86). They are deliberately centralized here so
// they can be retuned per GPU without touching kernel code. General rule: the
// right value scales with the GPU's SM count — more SMs want more parallel
// blocks (higher ATTN_PARTITIONS / ARGMAX_BLOCKS) to stay saturated.
//
//   SM counts for reference:  RTX 2070 ~36 | RTX 3090 ~82 | RTX 4090 ~128 |
//                             RTX 5090 ~170
// ===========================================================================

// ATTN_PARTITIONS — number of key-range splits in the flash-decoding attention
// (stage 1 launches n_kv_heads * ATTN_PARTITIONS blocks). Higher = more blocks
// / more parallelism at long context, but more redundant Q reloads, a larger
// stage-2 reduction, and bigger partial buffers. Must stay small enough that a
// partition still holds several keys at the context depths you care about.
//   RTX 2070: 32   |   RTX 3090: 64   |   RTX 4090: 96   |   RTX 5090: 128
constexpr int ATTN_PARTITIONS = 64;

// KSPLIT_WARPS — warps per attention stage-1 block. Must be >= the largest
// kv_group (= n_heads / n_kv_heads) across the models you build; for Qwen2.5
// that maximum is 8 (3B), so 8 is the floor. Rarely needs changing per GPU.
constexpr int KSPLIT_WARPS    = 8;

constexpr int ATTN_KEYS_CAP =
    (((QwenConfig::max_seq_len + ATTN_PARTITIONS - 1) / ATTN_PARTITIONS) + 15) / 16 * 16;

// GEMV_WARPS_PER_BLOCK — warps per block in the "tall" GEMV kernels (QKV, MLP
// gate/up, logits); one warp computes one output row. These have many output
// rows, so 8 already launches plenty of blocks and gives full occupancy.
//   RTX 2070: 8    |   RTX 3090: 8    |   RTX 4090: 8    |   RTX 5090: 8..16
constexpr int GEMV_WARPS_PER_BLOCK = 8;

// GEMV_WIDE_WARPS_PER_BLOCK — warps per block for the two GEMV kernels whose
// output-row count is comparatively small but whose contraction (input) dim is
// large: o_proj (rows = dim) and down_proj (rows = dim, input = intermediate).
// Packing more warps per block improves L2 reuse of the shared input vector.
// This helps ONLY when `dim` is large enough that 16 warps still launch enough
// blocks to fill the GPU: for the big model (dim >= 3072, i.e. 7B) it lifts
// these kernels ~6%; for the smaller models it would underfill the device and
// slightly regress, so they keep 8. The threshold is the knob to retune per GPU
// (raise it on small-SM parts like the RTX 2070; it can be lowered toward 2048
// on very wide GPUs such as the RTX 5090).
constexpr int GEMV_WIDE_WARPS_PER_BLOCK = (QwenConfig::dim >= 3072) ? 16 : 8;

// ARGMAX_BLOCKS / ARGMAX_THREADS — parallel greedy-sampling (argmax) reduction
// over the vocabulary. Stage 1 launches ARGMAX_BLOCKS blocks of ARGMAX_THREADS
// threads; stage 2 reduces the ARGMAX_BLOCKS partials with a single block of
// ARGMAX_BLOCKS threads. Scale ARGMAX_BLOCKS with SM count (keep it a power of
// two and <= 1024 so the stage-2 block can hold it). ARGMAX_THREADS: 256.
//   RTX 2070: 64   |   RTX 3090: 128  |   RTX 4090: 256  |   RTX 5090: 256
constexpr int ARGMAX_BLOCKS  = 128;
constexpr int ARGMAX_THREADS = 256;

#endif // CONFIG_H
