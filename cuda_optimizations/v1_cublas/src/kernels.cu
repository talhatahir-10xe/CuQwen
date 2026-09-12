#include "kernels.cuh"
#include <cmath>

// Embedding Lookup Kernel
__global__ void embedding_lookup_kernel(const half* embed_table, int token_id, half* output, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        output[idx] = embed_table[token_id * dim + idx];
    }
}

void launch_embedding_lookup(const half* embed_table, int token_id, half* output, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(embed_table, token_id, output, dim);
    CUDA_CHECK(cudaGetLastError());
}

// RMSNorm Kernel
__global__ void rmsnorm_kernel(const half* x, const half* weight, half* output, int dim, float eps) {
    int idx = threadIdx.x;
    
    float sum_sq = 0.0f;
    for (int i = idx; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    extern __shared__ float s_mem[];
    s_mem[idx] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (idx < stride) {
            s_mem[idx] += s_mem[idx + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf((s_mem[0] / static_cast<float>(dim)) + eps);

    for (int i = idx; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        float w = __half2float(weight[i]);
        output[i] = __float2half(val * inv_rms * w);
    }
}

void launch_rmsnorm(const half* x, const half* weight, half* output, int dim, float eps, cudaStream_t stream) {
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);
    rmsnorm_kernel<<<1, threads, shared_mem, stream>>>(x, weight, output, dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

// Add Bias Kernel
__global__ void add_bias_kernel(half* x, const half* bias, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        x[idx] = __hadd(x[idx], bias[idx]);
    }
}

void launch_add_bias(half* x, const half* bias, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(x, bias, dim);
    CUDA_CHECK(cudaGetLastError());
}

// Rotary Position Embedding (RoPE) Kernel
__global__ void rope_kernel(half* q, half* k, int pos, int n_heads, int n_kv_heads, int head_dim, float theta) {
    int head = blockIdx.x;
    int i = threadIdx.x;

    if (i >= head_dim / 2) return;

    float freq = 1.0f / powf(theta, (2.0f * i) / static_cast<float>(head_dim));
    float val = static_cast<float>(pos) * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    if (head < n_heads) {
        int idx1 = head * head_dim + i;
        int idx2 = head * head_dim + i + (head_dim / 2);

        float q1 = __half2float(q[idx1]);
        float q2 = __half2float(q[idx2]);

        q[idx1] = __float2half(q1 * cos_val - q2 * sin_val);
        q[idx2] = __float2half(q1 * sin_val + q2 * cos_val);
    }

    if (head < n_kv_heads) {
        int idx1 = head * head_dim + i;
        int idx2 = head * head_dim + i + (head_dim / 2);

        float k1 = __half2float(k[idx1]);
        float k2 = __half2float(k[idx2]);

        k[idx1] = __float2half(k1 * cos_val - k2 * sin_val);
        k[idx2] = __float2half(k1 * sin_val + k2 * cos_val);
    }
}

void launch_rope(half* q, half* k, int pos, int n_heads, int n_kv_heads, int head_dim, float theta, cudaStream_t stream) {
    int threads = head_dim / 2;
    int blocks = n_heads;
    rope_kernel<<<blocks, threads, 0, stream>>>(q, k, pos, n_heads, n_kv_heads, head_dim, theta);
    CUDA_CHECK(cudaGetLastError());
}

// SwiGLU Activation Kernel
__global__ void swiglu_kernel(const half* gate, const half* up, half* output, int inter_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inter_dim) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float swish = g * (1.0f / (1.0f + expf(-g)));
        output[idx] = __float2half(swish * u);
    }
}

void launch_swiglu(const half* gate, const half* up, half* output, int inter_dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = (inter_dim + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, inter_dim);
    CUDA_CHECK(cudaGetLastError());
}

// Elementwise Vector Addition Kernel
__global__ void add_kernel(const half* a, const half* b, half* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = __hadd(a[idx], b[idx]);
    }
}

void launch_add(const half* a, const half* b, half* out, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, size);
    CUDA_CHECK(cudaGetLastError());
}

// Grouped-Query Attention (GQA) Kernel (Fixed FP32 Accumulation)
__global__ void gqa_attention_kernel(
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
    int max_seq_len
) {
    int h = blockIdx.x;
    if (h >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = h / gqa_ratio;
    int kv_dim = n_kv_heads * head_dim;

    const half* q_head = q + h * head_dim;
    half* head_scores = att_scores + h * max_seq_len;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // 1. Compute Q * K^T Attention Scores (Accumulated in FP32)
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const half* k_ptr = key_cache + layer_offset + t * kv_dim + kv_head * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += __half2float(q_head[d]) * __half2float(k_ptr[d]);
        }
        head_scores[t] = __float2half(score * scale);
    }
    __syncthreads();

    // 2. Softmax Max-Reduction
    float max_val = -1e30f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float val = __half2float(head_scores[t]);
        if (val > max_val) max_val = val;
    }
    
    extern __shared__ float s_mem[];
    s_mem[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_mem[threadIdx.x] = fmaxf(s_mem[threadIdx.x], s_mem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    max_val = s_mem[0];

    // 3. Exponentiation & Sum Reduction
    float exp_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float e = expf(__half2float(head_scores[t]) - max_val);
        head_scores[t] = __float2half(e);
        exp_sum += e;
    }

    s_mem[threadIdx.x] = exp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / (s_mem[0] + 1e-10f);

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float norm_s = __half2float(head_scores[t]) * inv_sum;
        head_scores[t] = __float2half(norm_s);
    }
    __syncthreads();

    // 4. Compute Weighted Sum over Values V in FP32
    half* out_head = att_out + h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val_acc = 0.0f;
        for (int t = 0; t <= pos; ++t) {
            const half* v_ptr = value_cache + layer_offset + t * kv_dim + kv_head * head_dim;
            val_acc += __half2float(head_scores[t]) * __half2float(v_ptr[d]);
        }
        out_head[d] = __float2half(val_acc);
    }
}

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
    cudaStream_t stream
) {
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);
    gqa_attention_kernel<<<n_heads, threads, shared_mem, stream>>>(
        q, key_cache, value_cache, att_out, att_scores,
        pos, n_heads, n_kv_heads, head_dim, layer_offset, max_seq_len
    );
    CUDA_CHECK(cudaGetLastError());
}

// Compute Logits Kernel
__global__ void compute_logits_kernel(const half* norm_x, const half* lm_head_weight, half* logits, int vocab_size, int dim) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < vocab_size) {
        const half* w_row = lm_head_weight + v * dim;
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) {
            dot += __half2float(norm_x[d]) * __half2float(w_row[d]);
        }
        logits[v] = __float2half(dot);
    }
}

void launch_compute_logits(
    const half* norm_x,
    const half* lm_head_weight,
    half* logits,
    int vocab_size,
    int dim,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (vocab_size + threads - 1) / threads;
    compute_logits_kernel<<<blocks, threads, 0, stream>>>(norm_x, lm_head_weight, logits, vocab_size, dim);
    CUDA_CHECK(cudaGetLastError());
}

// GPU Repetition Penalty Kernel
__global__ void repetition_penalty_kernel(
    half* logits,
    const int* d_history,
    int history_len,
    int vocab_size,
    float penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < history_len) {
        int token_id = d_history[idx];
        if (token_id >= 0 && token_id < vocab_size) {
            float logit_val = __half2float(logits[token_id]);
            float res = (logit_val < 0.0f) ? (logit_val * penalty) : (logit_val / penalty);
            logits[token_id] = __float2half(res);
        }
    }
}

void launch_apply_repetition_penalty(
    half* logits,
    const int* d_history,
    int history_len,
    int vocab_size,
    float penalty,
    cudaStream_t stream
) {
    if (history_len <= 0) return;

    int threads = 256;
    int blocks = (history_len + threads - 1) / threads;
    repetition_penalty_kernel<<<blocks, threads, 0, stream>>>(logits, d_history, history_len, vocab_size, penalty);
    CUDA_CHECK(cudaGetLastError());
}

// cuBLAS Matrix-Vector Multiplication Wrapper
void matvec_gemv(cublasHandle_t handle, const half* W, const half* x, half* y, int rows, int cols, cudaStream_t stream) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        rows, 1, cols,
        &alpha,
        W, CUDA_R_16F, cols,
        x, CUDA_R_16F, cols,
        &beta,
        y, CUDA_R_16F, rows,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT
    ));
}

// GPU Argmax Kernel
__global__ void argmax_kernel(const half* logits, int* out_token, int vocab_size) {
    extern __shared__ float s_max_vals[];
    int* s_max_indices = (int*)&s_max_vals[blockDim.x];

    int tid = threadIdx.x;
    float thread_max_val = -1e30f;
    int thread_max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]);
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = i;
        }
    }

    s_max_vals[tid] = thread_max_val;
    s_max_indices[tid] = thread_max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max_vals[tid + stride] > s_max_vals[tid]) {
                s_max_vals[tid] = s_max_vals[tid + stride];
                s_max_indices[tid] = s_max_indices[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_token = s_max_indices[0];
    }
}

void launch_argmax(const half* logits, int* d_sampled_token, int vocab_size, cudaStream_t stream) {
    int threads = 256;
    size_t shared_mem = threads * (sizeof(float) + sizeof(int));
    argmax_kernel<<<1, threads, shared_mem, stream>>>(logits, d_sampled_token, vocab_size);
    CUDA_CHECK(cudaGetLastError());
}