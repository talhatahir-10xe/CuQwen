#include "model.cuh"
#include <cmath>
#include <iostream>

void malloc_qwen_state(const QwenConfig& config, QwenState& state) {
    const size_t dim           = config.dim;
    const size_t q_dim         = (size_t)config.n_heads * config.head_dim;
    const size_t kv_dim        = (size_t)config.n_kv_heads * config.head_dim;
    const size_t inter         = config.intermediate_size;
    const size_t kv_cache_size = (size_t)config.n_layers * config.max_seq_len * kv_dim;

    CUDA_CHECK(cudaMalloc(&state.x, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.xb, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.q, q_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.k, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.v, kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.att, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.gate, inter * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.up, inter * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.mlp_out, dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.logits, config.vocab_size * sizeof(half)));

    // Allocate GPU sampled token output buffer
    CUDA_CHECK(cudaMalloc(&state.d_sampled_token, sizeof(int)));

    // Device History Buffer Allocation
    CUDA_CHECK(cudaMalloc(&state.d_history, config.max_seq_len * sizeof(int)));

    // KV Cache Allocations
    CUDA_CHECK(cudaMalloc(&state.key_cache, kv_cache_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.value_cache, kv_cache_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(state.key_cache, 0, kv_cache_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(state.value_cache, 0, kv_cache_size * sizeof(half)));

    // Attention Score Buffer
    CUDA_CHECK(cudaMalloc(&state.att_scores, (size_t)config.n_heads * config.max_seq_len * sizeof(half)));
}

void free_qwen_state(QwenState& state) {
    CUDA_CHECK(cudaFree(state.x));
    CUDA_CHECK(cudaFree(state.xb));
    CUDA_CHECK(cudaFree(state.q));
    CUDA_CHECK(cudaFree(state.k));
    CUDA_CHECK(cudaFree(state.v));
    CUDA_CHECK(cudaFree(state.att));
    CUDA_CHECK(cudaFree(state.gate));
    CUDA_CHECK(cudaFree(state.up));
    CUDA_CHECK(cudaFree(state.mlp_out));
    CUDA_CHECK(cudaFree(state.logits));
    CUDA_CHECK(cudaFree(state.d_sampled_token));
    CUDA_CHECK(cudaFree(state.d_history));
    CUDA_CHECK(cudaFree(state.key_cache));
    CUDA_CHECK(cudaFree(state.value_cache));
    CUDA_CHECK(cudaFree(state.att_scores));
}

int qwen_forward(
    int token,
    int pos,
    const QwenConfig& config,
    QwenState& state,
    const QwenWeights& weights,
    cublasHandle_t handle,
    const int* h_history,
    int history_len,
    cudaStream_t stream
) {
    const int dim       = config.dim;
    const int q_dim     = config.n_heads * config.head_dim;
    const int kv_dim    = config.n_kv_heads * config.head_dim;
    const int inter     = config.intermediate_size;
    const int kv_stride = config.max_seq_len * kv_dim;

    // 1. Initial Embedding Lookup
    launch_embedding_lookup(weights.embed_tokens, token, state.x, dim, stream);

    // 2. Transformer Layers Execution Loop
    for (int l = 0; l < config.n_layers; ++l) {
        const LayerWeights& lw = weights.layers[l];

        launch_rmsnorm(state.x, lw.input_layernorm_weight, state.xb, dim, config.norm_eps, stream);

        matvec_gemv(handle, lw.q_proj_weight, state.xb, state.q, q_dim, dim, stream);
        matvec_gemv(handle, lw.k_proj_weight, state.xb, state.k, kv_dim, dim, stream);
        matvec_gemv(handle, lw.v_proj_weight, state.xb, state.v, kv_dim, dim, stream);

        launch_add_bias(state.q, lw.q_proj_bias, q_dim, stream);
        launch_add_bias(state.k, lw.k_proj_bias, kv_dim, stream);
        launch_add_bias(state.v, lw.v_proj_bias, kv_dim, stream);

        launch_rope(state.q, state.k, pos, config.n_heads, config.n_kv_heads, config.head_dim, config.rope_theta, stream);

        int layer_offset = l * kv_stride;
        half* k_dst = state.key_cache + layer_offset + pos * kv_dim;
        half* v_dst = state.value_cache + layer_offset + pos * kv_dim;
        CUDA_CHECK(cudaMemcpyAsync(k_dst, state.k, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(v_dst, state.v, kv_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream));

        launch_gqa_attention(
            state.q, state.key_cache, state.value_cache, state.xb, state.att_scores,
            pos, config.n_heads, config.n_kv_heads, config.head_dim, layer_offset, config.max_seq_len, stream
        );

        matvec_gemv(handle, lw.o_proj_weight, state.xb, state.att, dim, q_dim, stream);
        launch_add(state.x, state.att, state.x, dim, stream);

        launch_rmsnorm(state.x, lw.post_attention_layernorm_weight, state.xb, dim, config.norm_eps, stream);
        matvec_gemv(handle, lw.gate_proj_weight, state.xb, state.gate, inter, dim, stream);
        matvec_gemv(handle, lw.up_proj_weight, state.xb, state.up, inter, dim, stream);
        launch_swiglu(state.gate, state.up, state.gate, inter, stream);
        matvec_gemv(handle, lw.down_proj_weight, state.gate, state.mlp_out, dim, inter, stream);

        launch_add(state.x, state.mlp_out, state.x, dim, stream);
    }

    // 3. Final RMSNorm & Compute Raw Logits
    launch_rmsnorm(state.x, weights.norm_weight, state.xb, dim, config.norm_eps, stream);
    launch_compute_logits(state.xb, weights.lm_head_weight, state.logits, config.vocab_size, dim, stream);

    // 4. Apply Repetition Penalty Directly on GPU Logits
    if (h_history != nullptr && history_len > 0) {
        CUDA_CHECK(cudaMemcpyAsync(
            state.d_history,
            h_history,
            history_len * sizeof(int),
            cudaMemcpyHostToDevice,
            stream
        ));
        launch_apply_repetition_penalty(
            state.logits,
            state.d_history,
            history_len,
            config.vocab_size,
            config.repetition_penalty,
            stream
        );
    }

    // 5. GPU Sampling (Argmax)
    launch_argmax(state.logits, state.d_sampled_token, config.vocab_size, stream);

    // 6. Copy back sampled token ID integer
    int h_next_token = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_next_token, state.d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return h_next_token;
}