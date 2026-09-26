#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "config.h"
// ATTN_PARTITIONS and HEADS_PER_WARP tuning parameters live in config.h.

// Head slots reserved per KV-head in the FlashDecoding partial buffers. It is
// >= any supported kv_group (n_heads/n_kv_heads) and equals the 16-row WMMA
// fragment height, so the tensor-core stage-1 can store its O fragments (16
// rows: kv_group real query heads + padding) straight to global memory with no
// shared-memory round-trip. Partial index for (kv_head, local head) is
// (kv_head*PARTIAL_HEAD_SLOTS + local_head)*ATTN_PARTITIONS + partition.
constexpr int PARTIAL_HEAD_SLOTS = 16;


void launch_embedding_lookup(const half* embed_table, const int* d_token_id, half* output, int dim, cudaStream_t stream = 0);

void launch_gqa_attention_decode(
    const half* q, 
    const half* key_cache, 
    const half* value_cache, 
    half* attn_out,
    const int* d_pos, 
    int n_heads, 
    int n_kv_heads, 
    int head_dim,
    int max_seq_len, 
    float* d_partial_out, 
    float* d_partial_max,
    float* d_partial_sum, 
    cudaStream_t stream = 0
);

void launch_kv_cache_store(
    const half* k_src,
    const half* v_src,
    half* layer_k_cache,
    half* layer_v_cache,
    const int* d_pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim,
    cudaStream_t stream = 0
);

void launch_apply_repetition_penalty(
    half* logits,
    const int* d_history,
    const int* d_history_len,
    int vocab_size,
    half penalty,
    cudaStream_t stream = 0
);

void launch_argmax(
    const half* logits,
    int* d_sampled_token,
    int vocab_size,
    cudaStream_t stream = 0
);

// Note on `qweight_t` + `*_scale` params below: qweight_t == half for FP16
// builds and int8_t for INT8 (W8A16) builds. The scale pointers carry one FP16
// scale per QUANT_GROUP_SIZE weights and are used only in INT8 builds (pass
// nullptr for FP16). See load_w8() in kernels.cu for the in-kernel dequant.
void launch_fused_attn_block(
    const half* x,
    const half* norm_weight,
    const qweight_t* W_q,
    const qweight_t* W_k,
    const qweight_t* W_v,
    const half* W_q_scale,
    const half* W_k_scale,
    const half* W_v_scale,
    const half* b_q,
    const half* b_k,
    const half* b_v,
    half* q_out,
    half* k_out,
    half* v_out,
    int dim,
    int q_dim,
    int kv_dim,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    const int* d_pos,
    float rope_theta,
    half eps,
    cudaStream_t stream = 0
);

void launch_fused_mlp_stage1(
    const half* x,
    const half* norm_weight,
    const qweight_t* gate_weight,
    const qweight_t* up_weight,
    const half* gate_scale,
    const half* up_scale,
    half* intermediate_out,
    int dim,
    int inter_dim,
    half eps,
    cudaStream_t stream = 0
);

void launch_fused_mlp_stage2(
    half* x,
    const qweight_t* down_weight,
    const half* down_scale,
    const half* intermediate_in,
    int dim,
    int inter_dim,
    cudaStream_t stream = 0
);

// GEMV + residual add (used for o_proj). Retains the historical name; under
// INT8 the weight is dequantized in-kernel via load_w8().
void launch_gemv_add_fp16(
    const qweight_t* W,
    const half* W_scale,
    const half* input,
    half* x_inout,
    int rows,
    int cols,
    cudaStream_t stream = 0
);

void launch_rmsnorm(
    const half* x,
    const half* norm_weight,
    half* out,
    int dim,
    half eps,
    cudaStream_t stream = 0
);

void launch_compute_logits(
    const half* x_normed,
    const half* lm_head_weight,
    half* logits,
    int vocab_size,
    int dim,
    cudaStream_t stream = 0
);

#endif // KERNELS_CUH