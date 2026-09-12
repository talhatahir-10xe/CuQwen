#ifndef MODEL_CUH
#define MODEL_CUH

#include <cuda_fp16.h>
#include "config.h"
#include "kernels.cuh"
#include "load_weights.h"

struct QwenState {
    half* x;
    half* xb;
    half* q;
    half* k;
    half* v;
    half* att;
    half* gate;
    half* up;
    half* mlp_out;
    half* logits;

    // Device buffer for GPU sampled token
    int* d_sampled_token;

    // GPU History Buffer for Repetition Penalty
    int* d_history;

    // Head-Major KV Cache: [n_layers, n_kv_heads, max_seq_len, head_dim]
    half* key_cache;
    half* value_cache;

    // FlashDecoding Work Buffers
    float* d_partial_out;
    float* d_partial_max;
    float* d_partial_sum;

    // Dynamic inputs for CUDA Graph execution
    int* d_token;
    int* d_pos;
    int* d_history_len;

    // CUDA Graph objects & Stream
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    bool graph_created;
};

void malloc_qwen_state(const QwenConfig& config, QwenState& state);
void free_qwen_state(QwenState& state);

int qwen_forward(
    int token,
    int pos,
    const QwenConfig& config,
    QwenState& state,
    const QwenWeights& weights,
    const int* h_history = nullptr,
    int history_len = 0,
    cudaStream_t stream = 0
);

#endif // MODEL_CUH