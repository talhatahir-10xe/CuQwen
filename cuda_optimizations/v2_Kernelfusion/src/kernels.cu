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

// Grouped-Query Attention (GQA)
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

    // 1. Compute Q * K^T Attention Scores in FP32
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
    float max_val = -1e9f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float val = __half2float(head_scores[t]);
        if (val > max_val) max_val = val;
    }

    extern __shared__ float s_mem_fp32[];
    s_mem_fp32[threadIdx.x] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            if (s_mem_fp32[threadIdx.x + stride] > s_mem_fp32[threadIdx.x]) {
                s_mem_fp32[threadIdx.x] = s_mem_fp32[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    max_val = s_mem_fp32[0];

    // 3. Exponentiation & Sum Reduction
    float exp_sum = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float e = expf(__half2float(head_scores[t]) - max_val);
        head_scores[t] = __float2half(e);
        exp_sum += e;
    }

    s_mem_fp32[threadIdx.x] = exp_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s_mem_fp32[threadIdx.x] += s_mem_fp32[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / (s_mem_fp32[0] + 1e-5f);

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        head_scores[t] = __float2half(__half2float(head_scores[t]) * inv_sum);
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

// Fused Attention Block Kernel (RMSNorm + Parallel GEMV + RoPE)
__global__ void fused_attn_block_kernel(
    const half* __restrict__ x,
    const half* __restrict__ norm_weight,
    const half* __restrict__ W_q,
    const half* __restrict__ W_k,
    const half* __restrict__ W_v,
    const half* __restrict__ b_q,
    const half* __restrict__ b_k,
    const half* __restrict__ b_v,
    half* __restrict__ q_out,
    half* __restrict__ k_out,
    half* __restrict__ v_out,
    int dim,
    int q_dim,
    int kv_dim,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float rope_theta,
    half eps
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    extern __shared__ float s_mem_fp32[];
    float* s_reduce = s_mem_fp32;

    // 1. FP32 Shared RMSNorm Reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_reduce[0] / static_cast<float>(dim);
    float inv_rms_f32 = rsqrtf(mean_sq + __half2float(eps));
    __syncthreads();

    // 2. Identify Target Block
    const half* W_row = nullptr;
    float bias_val = 0.0f;
    int target_idx = 0;
    enum TensorType { TARGET_Q = 0, TARGET_K = 1, TARGET_V = 2 } target_type;

    if (bid < q_dim) {
        target_type = TARGET_Q;
        target_idx = bid;
        W_row = W_q + target_idx * dim;
        bias_val = __half2float(b_q[target_idx]);
    } else if (bid < q_dim + kv_dim) {
        target_type = TARGET_K;
        target_idx = bid - q_dim;
        W_row = W_k + target_idx * dim;
        bias_val = __half2float(b_k[target_idx]);
    } else {
        target_type = TARGET_V;
        target_idx = bid - q_dim - kv_dim;
        W_row = W_v + target_idx * dim;
        bias_val = __half2float(b_v[target_idx]);
    }

    // 3. Parallel FP32 Matrix-Vector Dot Product
    float thread_acc = 0.0f;
    for (int c = tid; c < dim; c += blockDim.x) {
        float x_normed = __half2float(x[c]) * inv_rms_f32 * __half2float(norm_weight[c]);
        thread_acc += __half2float(W_row[c]) * x_normed;
    }

    s_reduce[tid] = thread_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    // 4. Output Writing & Pairwise RoPE Transformation
    if (tid == 0) {
        float projected_val = s_reduce[0] + bias_val;

        if (target_type == TARGET_V) {
            v_out[target_idx] = __float2half(projected_val);
        } else {
            int feature_idx = target_idx % head_dim;
            int half_hd = head_dim / 2;
            int rot_i = feature_idx % half_hd;

            float freq = 1.0f / powf(rope_theta, (2.0f * rot_i) / static_cast<float>(head_dim));
            float val = static_cast<float>(pos) * freq;
            float cos_v = cosf(val);
            float sin_v = sinf(val);

            // Compute Pair Element Index
            int pair_feature_idx = (feature_idx < half_hd) ? (feature_idx + half_hd) : (feature_idx - half_hd);
            int head_i = target_idx / head_dim;
            int pair_target_idx = head_i * head_dim + pair_feature_idx;

            const half* W_pair_row = (target_type == TARGET_Q) ? (W_q + pair_target_idx * dim) : (W_k + pair_target_idx * dim);
            float pair_bias = (target_type == TARGET_Q) ? __half2float(b_q[pair_target_idx]) : __half2float(b_k[pair_target_idx]);

            float pair_acc = 0.0f;
            for (int c = 0; c < dim; ++c) {
                float x_normed = __half2float(x[c]) * inv_rms_f32 * __half2float(norm_weight[c]);
                pair_acc += __half2float(W_pair_row[c]) * x_normed;
            }
            float pair_val = pair_acc + pair_bias;

            float rotated_res;
            if (feature_idx < half_hd) {
                rotated_res = projected_val * cos_v - pair_val * sin_v;
            } else {
                rotated_res = pair_val * sin_v + projected_val * cos_v;
            }

            if (target_type == TARGET_Q) {
                q_out[target_idx] = __float2half(rotated_res);
            } else {
                k_out[target_idx] = __float2half(rotated_res);
            }
        }
    }
}

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
    int pos,
    float rope_theta,
    half eps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = q_dim + (2 * kv_dim);
    size_t shared_mem = threads * sizeof(float);

    fused_attn_block_kernel<<<blocks, threads, shared_mem, stream>>>(
        x, norm_weight, W_q, W_k, W_v, b_q, b_k, b_v,
        q_out, k_out, v_out, dim, q_dim, kv_dim,
        n_heads, n_kv_heads, head_dim, pos, rope_theta, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused MLP Stage 1 Kernel (RMSNorm + SwiGLU)
__global__ void fused_mlp_stage1_kernel(
    const half* __restrict__ x,
    const half* __restrict__ norm_weight,
    const half* __restrict__ gate_weight,
    const half* __restrict__ up_weight,
    half* __restrict__ intermediate_out,
    int dim,
    int inter_dim,
    half eps
) {
    int tid = threadIdx.x;
    int row = blockIdx.x;

    if (row >= inter_dim) return;

    extern __shared__ float s_mlp_mem_fp32[];
    float* s_reduce = s_mlp_mem_fp32;

    // 1. FP32 Shared RMSNorm Reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_reduce[0] / static_cast<float>(dim);
    float inv_rms_f32 = rsqrtf(mean_sq + __half2float(eps));
    __syncthreads();

    // 2. Gate & Up GEMV Reductions
    const half* gate_row = gate_weight + row * dim;
    const half* up_row   = up_weight + row * dim;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (int c = tid; c < dim; c += blockDim.x) {
        float x_normed = __half2float(x[c]) * inv_rms_f32 * __half2float(norm_weight[c]);
        gate_acc += __half2float(gate_row[c]) * x_normed;
        up_acc   += __half2float(up_row[c]) * x_normed;
    }

    s_reduce[tid] = gate_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float gate_total = s_reduce[0];
    __syncthreads();

    s_reduce[tid] = up_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce[tid] += s_reduce[tid + stride];
        }
        __syncthreads();
    }
    float up_total = s_reduce[0];

    // 3. SwiGLU Activation
    if (tid == 0) {
        float sigmoid_g = 1.0f / (1.0f + expf(-gate_total));
        float swish = gate_total * sigmoid_g;
        intermediate_out[row] = __float2half(swish * up_total);
    }
}

void launch_fused_mlp_stage1(
    const half* x,
    const half* norm_weight,
    const half* gate_weight,
    const half* up_weight,
    half* intermediate_out,
    int dim,
    int inter_dim,
    half eps,
    cudaStream_t stream
) {
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);

    fused_mlp_stage1_kernel<<<inter_dim, threads, shared_mem, stream>>>(
        x, norm_weight, gate_weight, up_weight, intermediate_out, dim, inter_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused MLP Stage 2 (Down GEMV + Residual Add in FP32 Accumulation)
__global__ void fused_mlp_stage2_kernel(
    half* __restrict__ x,
    const half* __restrict__ down_weight,
    const half* __restrict__ intermediate_in,
    int dim,
    int inter_dim
) {
    int row = blockIdx.x;
    if (row >= dim) return;

    int tid = threadIdx.x;
    const half* down_row = down_weight + row * inter_dim;

    extern __shared__ float s_mlp_stage2_mem[];
    float thread_acc = 0.0f;

    for (int c = tid; c < inter_dim; c += blockDim.x) {
        thread_acc += __half2float(down_row[c]) * __half2float(intermediate_in[c]);
    }

    s_mlp_stage2_mem[tid] = thread_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mlp_stage2_mem[tid] += s_mlp_stage2_mem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float x_orig = __half2float(x[row]);
        x[row] = __float2half(x_orig + s_mlp_stage2_mem[0]);
    }
}

void launch_fused_mlp_stage2(
    half* x,
    const half* down_weight,
    const half* intermediate_in,
    int dim,
    int inter_dim,
    cudaStream_t stream
) {
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);

    fused_mlp_stage2_kernel<<<dim, threads, shared_mem, stream>>>(
        x, down_weight, intermediate_in, dim, inter_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused GEMV + Residual Add Kernel
__global__ void gemv_add_fp16_kernel(
    const half* __restrict__ W,
    const half* __restrict__ input,
    half* __restrict__ x_inout,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    const half* W_row = W + row * cols;

    extern __shared__ float s_gemv_mem[];
    float thread_acc = 0.0f;

    for (int c = tid; c < cols; c += blockDim.x) {
        thread_acc += __half2float(W_row[c]) * __half2float(input[c]);
    }

    s_gemv_mem[tid] = thread_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_gemv_mem[tid] += s_gemv_mem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float orig_x = __half2float(x_inout[row]);
        x_inout[row] = __float2half(orig_x + s_gemv_mem[0]);
    }
}

void launch_gemv_add_fp16(
    const half* W,
    const half* input,
    half* x_inout,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int threads = 256;
    size_t shared_mem = threads * sizeof(float);

    gemv_add_fp16_kernel<<<rows, threads, shared_mem, stream>>>(W, input, x_inout, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// Standalone RMSNorm
__global__ void rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ norm_weight,
    half* __restrict__ out,
    int dim,
    half eps
) {
    int tid = threadIdx.x;
    extern __shared__ float s_mem[];

    // 1. FP32 Shared RMSNorm Reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }
    s_mem[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_mem[0] / static_cast<float>(dim);
    float inv_rms_f32 = rsqrtf(mean_sq + __half2float(eps));
    __syncthreads();

    // 2. Write normalized and scaled values
    for (int i = tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]) * inv_rms_f32 * __half2float(norm_weight[i]);
        out[i] = __float2half(val);
    }
}

void launch_rmsnorm(
    const half* x,
    const half* norm_weight,
    half* out,
    int dim,
    half eps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = 1;
    size_t shared_mem = threads * sizeof(float);

    rmsnorm_kernel<<<blocks, threads, shared_mem, stream>>>(
        x, norm_weight, out, dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

//  Logits GEMV Kernel
__global__ void compute_logits_kernel(
    const half* __restrict__ x_normed,
    const half* __restrict__ lm_head_weight,
    half* __restrict__ logits,
    int vocab_size,
    int dim
) {
    int v = blockIdx.x;
    if (v >= vocab_size) return;

    int tid = threadIdx.x;
    extern __shared__ float s_logits_mem[];

    const half* w_row = lm_head_weight + static_cast<size_t>(v) * dim;
    float dot_acc = 0.0f;

    for (int d = tid; d < dim; d += blockDim.x) {
        dot_acc += __half2float(x_normed[d]) * __half2float(w_row[d]);
    }

    s_logits_mem[tid] = dot_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_logits_mem[tid] += s_logits_mem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        logits[v] = __float2half(s_logits_mem[0]);
    }
}

void launch_compute_logits(
    const half* x_normed,
    const half* lm_head_weight,
    half* logits,
    int vocab_size,
    int dim,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = vocab_size;
    size_t shared_mem = threads * sizeof(float);

    compute_logits_kernel<<<blocks, threads, shared_mem, stream>>>(
        x_normed, lm_head_weight, logits, vocab_size, dim
    );
    CUDA_CHECK(cudaGetLastError());
}

// GPU Repetition Penalty Kernel
__global__ void repetition_penalty_kernel(
    half* logits,
    const int* d_history,
    int history_len,
    int vocab_size,
    half penalty
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < history_len) {
        int token_id = d_history[idx];
        if (token_id >= 0 && token_id < vocab_size) {
            float logit_val = __half2float(logits[token_id]);
            float pen = __half2float(penalty);
            float res = (logit_val < 0.0f) ? (logit_val * pen) : (logit_val / pen);
            logits[token_id] = __float2half(res);
        }
    }
}

void launch_apply_repetition_penalty(
    half* logits,
    const int* d_history,
    int history_len,
    int vocab_size,
    half penalty,
    cudaStream_t stream
) {
    if (history_len <= 0) return;

    int threads = 256;
    int blocks = (history_len + threads - 1) / threads;
    repetition_penalty_kernel<<<blocks, threads, 0, stream>>>(logits, d_history, history_len, vocab_size, penalty);
    CUDA_CHECK(cudaGetLastError());
}

// GPU Argmax Kernel
__global__ void argmax_kernel(const half* logits, int* out_token, int vocab_size) {
    extern __shared__ float s_max_vals_fp32[];
    int* s_max_indices = (int*)&s_max_vals_fp32[blockDim.x];

    int tid = threadIdx.x;
    float thread_max_val = -1e9f;
    int thread_max_idx = 0;

    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]);
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = i;
        }
    }

    s_max_vals_fp32[tid] = thread_max_val;
    s_max_indices[tid] = thread_max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (s_max_vals_fp32[tid + stride] > s_max_vals_fp32[tid]) {
                s_max_vals_fp32[tid] = s_max_vals_fp32[tid + stride];
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