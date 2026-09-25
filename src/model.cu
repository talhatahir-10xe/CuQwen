#include "model.cuh"
#include <cmath>
#include <iostream>

void malloc_qwen_state(const QwenConfig& config, QwenState& state) {
    const size_t dim           = config.dim;
    const size_t q_dim         = (size_t)config.n_heads * config.head_dim;
    const size_t kv_dim        = (size_t)config.n_kv_heads * config.head_dim;
    const size_t inter         = config.intermediate_size;
    const size_t kv_cache_size = (size_t)config.n_layers * config.n_kv_heads * config.max_seq_len * config.head_dim;

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

    CUDA_CHECK(cudaMalloc(&state.d_sampled_token, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_history, config.max_seq_len * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&state.d_token, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_pos, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_history_len, sizeof(int)));

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    state.graph = nullptr;
    state.graph_exec = nullptr;
    state.graph_created = false;

    CUDA_CHECK(cudaMalloc(&state.key_cache, kv_cache_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&state.value_cache, kv_cache_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(state.key_cache, 0, kv_cache_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(state.value_cache, 0, kv_cache_size * sizeof(half)));

    const size_t required_splits = (size_t)config.n_kv_heads * PARTIAL_HEAD_SLOTS * ATTN_PARTITIONS;
    CUDA_CHECK(cudaMalloc(&state.d_partial_out, required_splits * config.head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_partial_max, required_splits * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state.d_partial_sum, required_splits * sizeof(float)));
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
    CUDA_CHECK(cudaFree(state.d_token));
    CUDA_CHECK(cudaFree(state.d_pos));
    CUDA_CHECK(cudaFree(state.d_history_len));
    CUDA_CHECK(cudaFree(state.key_cache));
    CUDA_CHECK(cudaFree(state.value_cache));
    CUDA_CHECK(cudaFree(state.d_partial_out));
    CUDA_CHECK(cudaFree(state.d_partial_max));
    CUDA_CHECK(cudaFree(state.d_partial_sum));

    if (state.graph_created) {
        CUDA_CHECK(cudaGraphExecDestroy(state.graph_exec));
        CUDA_CHECK(cudaGraphDestroy(state.graph));
        state.graph_created = false;
    }
    CUDA_CHECK(cudaStreamDestroy(state.stream));
}

static void qwen_forward_graph_body(
    const QwenConfig& config,
    QwenState& state,
    const QwenWeights& weights,
    cudaStream_t stream
) {
    const int dim       = config.dim;
    const int q_dim     = config.n_heads * config.head_dim;
    const int kv_dim    = config.n_kv_heads * config.head_dim;
    const int inter     = config.intermediate_size;
    const int head_dim  = config.head_dim;
    const int n_kv      = config.n_kv_heads;

    launch_embedding_lookup(weights.embed_tokens, state.d_token, state.x, dim, stream);

    for (int l = 0; l < config.n_layers; ++l) {
        const LayerWeights& lw = weights.layers[l];

        launch_rmsnorm(
            state.x, lw.input_layernorm_weight, state.att,
            dim, config.norm_eps, stream
        );

        launch_fused_attn_block(
            state.att,
            lw.input_layernorm_weight,
            lw.q_proj_weight, lw.k_proj_weight, lw.v_proj_weight,
            lw.q_proj_bias, lw.k_proj_bias, lw.v_proj_bias,
            state.q, state.k, state.v,
            dim, q_dim, kv_dim,
            config.n_heads, config.n_kv_heads, head_dim,
            state.d_pos, config.rope_theta, config.norm_eps,
            stream
        );

        size_t layer_kv_offset = (size_t)l * n_kv * config.max_seq_len * head_dim;
        half* layer_k_cache = state.key_cache + layer_kv_offset;
        half* layer_v_cache = state.value_cache + layer_kv_offset;

        // Write Key/Value into head-major KV cache slot dynamically via GPU kernel
        launch_kv_cache_store(
            state.k, state.v, layer_k_cache, layer_v_cache,
            state.d_pos, n_kv, config.max_seq_len, head_dim, stream
        );

        launch_gqa_attention_decode(
            state.q, layer_k_cache, layer_v_cache, state.xb,
            state.d_pos, config.n_heads, config.n_kv_heads, head_dim,
            config.max_seq_len, state.d_partial_out, state.d_partial_max,
            state.d_partial_sum, stream
        );

        launch_gemv_add_fp16(lw.o_proj_weight, state.xb, state.x, dim, q_dim, stream);

        launch_rmsnorm(
            state.x, lw.post_attention_layernorm_weight, state.att,
            dim, config.norm_eps, stream
        );

        launch_fused_mlp_stage1(
            state.att,
            lw.post_attention_layernorm_weight,
            lw.gate_proj_weight,
            lw.up_proj_weight,
            state.gate,
            dim, inter, config.norm_eps,
            stream
        );

        launch_fused_mlp_stage2(
            state.x,
            lw.down_proj_weight,
            state.gate,
            dim, inter,
            stream
        );
    }

    launch_rmsnorm(
        state.x, weights.norm_weight, state.xb,
        dim, config.norm_eps, stream
    );

    launch_compute_logits(
        state.xb, weights.lm_head_weight, state.logits,
        config.vocab_size, dim, stream
    );

    launch_apply_repetition_penalty(
        state.logits, state.d_history, state.d_history_len,
        config.vocab_size, config.repetition_penalty, stream
    );

    launch_argmax(state.logits, state.d_sampled_token, config.vocab_size, stream);
}

int qwen_forward(
    int token,
    int pos,
    const QwenConfig& config,
    QwenState& state,
    const QwenWeights& weights,
    const int* h_history,
    int history_len,
    cudaStream_t stream
) {
    cudaStream_t exec_stream = (stream != 0) ? stream : state.stream;

    CUDA_CHECK(cudaMemcpyAsync(state.d_token, &token, sizeof(int), cudaMemcpyHostToDevice, exec_stream));
    CUDA_CHECK(cudaMemcpyAsync(state.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice, exec_stream));
    CUDA_CHECK(cudaMemcpyAsync(state.d_history_len, &history_len, sizeof(int), cudaMemcpyHostToDevice, exec_stream));

    if (h_history != nullptr && history_len > 0) {
        CUDA_CHECK(cudaMemcpyAsync(state.d_history, h_history, history_len * sizeof(int), cudaMemcpyHostToDevice, exec_stream));
    }

    if (!state.graph_created) {
        qwen_forward_graph_body(config, state, weights, exec_stream);
        CUDA_CHECK(cudaStreamSynchronize(exec_stream));

        CUDA_CHECK(cudaStreamBeginCapture(exec_stream, cudaStreamCaptureModeGlobal));
        qwen_forward_graph_body(config, state, weights, exec_stream);
        CUDA_CHECK(cudaStreamEndCapture(exec_stream, &state.graph));

        CUDA_CHECK(cudaGraphInstantiate(&state.graph_exec, state.graph, NULL, NULL, 0));
        state.graph_created = true;
    }

    // Launch CUDA Graph
    CUDA_CHECK(cudaGraphLaunch(state.graph_exec, exec_stream));

    int h_next_token = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_next_token, state.d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost, exec_stream));
    CUDA_CHECK(cudaStreamSynchronize(exec_stream));

    return h_next_token;
}