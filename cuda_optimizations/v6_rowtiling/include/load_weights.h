#ifndef LOAD_WEIGHTS_H
#define LOAD_WEIGHTS_H

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <string>
#include "config.h"

// Weights contained within a single Qwen2.5 Transformer Layer
struct LayerWeights {
    half* input_layernorm_weight;          // [dim]
    half* q_proj_weight;                   // [q_dim, dim]
    half* q_proj_bias;                     // [q_dim]
    half* k_proj_weight;                   // [kv_dim, dim]
    half* k_proj_bias;                     // [kv_dim]
    half* v_proj_weight;                   // [kv_dim, dim]
    half* v_proj_bias;                     // [kv_dim]
    half* o_proj_weight;                   // [dim, q_dim]
    half* post_attention_layernorm_weight; // [dim]
    half* gate_proj_weight;                // [inter, dim]
    half* up_proj_weight;                  // [inter, dim]
    half* down_proj_weight;                // [dim, inter]
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