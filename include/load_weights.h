#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <string>
#include "config.h"

// Weights contained within a single Qwen2.5 Transformer Layer.
//
// The linear-projection weights use `qweight_t` (== half for FP16 builds,
// int8_t for INT8/W8A16 builds). In INT8 builds each of these matrices is
// paired with a `*_scale` buffer holding one FP16 scale per QUANT_GROUP_SIZE
// weights (grouped along the input/contraction dim); those pointers stay
// nullptr in FP16 builds. RMSNorm weights and biases always remain FP16.
struct LayerWeights {
    half*      input_layernorm_weight;          // [dim]
    qweight_t* q_proj_weight;                   // [q_dim, dim]
    half*      q_proj_bias;                      // [q_dim]
    qweight_t* k_proj_weight;                   // [kv_dim, dim]
    half*      k_proj_bias;                      // [kv_dim]
    qweight_t* v_proj_weight;                   // [kv_dim, dim]
    half*      v_proj_bias;                      // [kv_dim]
    qweight_t* o_proj_weight;                   // [dim, q_dim]
    half*      post_attention_layernorm_weight; // [dim]
    qweight_t* gate_proj_weight;                // [inter, dim]
    qweight_t* up_proj_weight;                  // [inter, dim]
    qweight_t* down_proj_weight;                // [dim, inter]

    // Per-group FP16 scales for the quantized matrices (nullptr in FP16 builds).
    half* q_proj_scale;
    half* k_proj_scale;
    half* v_proj_scale;
    half* o_proj_scale;
    half* gate_proj_scale;
    half* up_proj_scale;
    half* down_proj_scale;
};

// Container for all Qwen2.5 GPU Model Weights
struct QwenWeights {
    half* embed_tokens;                    // [vocab_size, dim]
    std::vector<LayerWeights> layers;      // [n_layers]
    half* norm_weight;                     // [dim]
    half* lm_head_weight;                  // Pointer to embed_tokens (tied embeddings)
};

// Loads binary weights into GPU VRAM and verifies configuration header
QwenConfig load_weights(const std::string& bin_path, QwenWeights& weights);

// Releases all allocated GPU memory allocations
void free_weights(QwenWeights& weights, const QwenConfig& config);

#endif // LOAD_WEIGHTS_H