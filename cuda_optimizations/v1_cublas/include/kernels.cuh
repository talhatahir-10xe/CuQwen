#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cublas_v2.h>
#include "config.h"

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error: " << status \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void launch_embedding_lookup(const half* embed_table, int token_id, half* output, int dim, cudaStream_t stream = 0);
void launch_rmsnorm(const half* x, const half* weight, half* output, int dim, float eps, cudaStream_t stream = 0);
void launch_add_bias(half* x, const half* bias, int dim, cudaStream_t stream = 0);
void launch_rope(half* q, half* k, int pos, int n_heads, int n_kv_heads, int head_dim, float theta, cudaStream_t stream = 0);
void launch_softmax(half* scores, int seq_len, float scale, cudaStream_t stream = 0);
void launch_swiglu(const half* gate, const half* up, half* output, int inter_dim, cudaStream_t stream = 0);
void launch_add(const half* a, const half* b, half* out, int size, cudaStream_t stream = 0);

void launch_gqa_attention(
    const half* q,
    const half* key_cache,
    const half* value_cache,
    half* att_out,
    half* att_scores,
    int pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int layer_offset,
    int max_seq_len,
    cudaStream_t stream = 0
);

void launch_compute_logits(
    const half* norm_x,
    const half* lm_head_weight,
    half* logits,
    int vocab_size,
    int dim,
    cudaStream_t stream = 0
);

void launch_apply_repetition_penalty(
    half* logits,
    const int* d_history,
    int history_len,
    int vocab_size,
    float penalty,
    cudaStream_t stream = 0
);

void launch_argmax(
    const half* logits,
    int* d_sampled_token,
    int vocab_size,
    cudaStream_t stream = 0
);

void matvec_gemv(cublasHandle_t handle, const half* W, const half* x, half* y, int rows, int cols, cudaStream_t stream = 0);

#endif // KERNELS_CUH