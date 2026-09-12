#ifndef MODEL_CUH
#define MODEL_CUH

#include <cuda_fp16.h>
#include <cublas_v2.h>
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

    // KV Cache
    half* key_cache;
    half* value_cache;

    // Attention Scores Buffer
    half* att_scores;
};

void malloc_qwen_state(const QwenConfig& config, QwenState& state);
void free_qwen_state(QwenState& state);

int qwen_forward(
    int token,
    int pos,
    const QwenConfig& config,
    QwenState& state,
    const QwenWeights& weights,
    cublasHandle_t handle,
    const int* h_history = nullptr,
    int history_len = 0,
    cudaStream_t stream = 0
);

#endif // MODEL_CUH