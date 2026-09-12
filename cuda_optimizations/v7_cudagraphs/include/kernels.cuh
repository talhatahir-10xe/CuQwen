#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "config.h"


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

void launch_fused_attn_block(
    const half* x,
    const half* norm_weight,
    const half* W_q,
    const half* W_k,
    const half* W_v,
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
    const half* gate_weight,
    const half* up_weight,
    half* intermediate_out,
    int dim,
    int inter_dim,
    half eps,
    cudaStream_t stream = 0
);

void launch_fused_mlp_stage2(
    half* x,
    const half* down_weight,
    const half* intermediate_in,
    int dim,
    int inter_dim,
    cudaStream_t stream = 0
);

void launch_gemv_add_fp16(
    const half* W,
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