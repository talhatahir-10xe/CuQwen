#include "kernels.cuh"
#include <cmath>
#include <cuda_fp16.h>

// Helper struct for 128-bit vectorized loading (8 x half = 16 bytes = 1 float4)
union Vector128 {
    float4 f4;
    half2 h2[4];
};

// Embedding Lookup Kernel
__global__ void embedding_lookup_kernel(const half* embed_table, int token_id, half* output, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_dim = dim / 8;

    const float4* embed_ptr = reinterpret_cast<const float4*>(embed_table + token_id * dim);
    float4* out_ptr = reinterpret_cast<float4*>(output);

    if (idx < vec_dim) {
        out_ptr[idx] = embed_ptr[idx];
    }
}

void launch_embedding_lookup(const half* embed_table, int token_id, half* output, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((dim / 8) + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(embed_table, token_id, output, dim);
    CUDA_CHECK(cudaGetLastError());
}

// Grouped-Query Attention (GQA) Kernel
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

    const half* q_head = q + h * head_dim;
    half* head_scores = att_scores + h * max_seq_len;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    size_t kv_head_offset = layer_offset + (size_t)kv_head * max_seq_len * head_dim;
    int vec_head_dim = head_dim / 8;

    // 1. Q * K^T Attention Scores (Vectorized over head_dim)
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        const float4* k_ptr_vec = reinterpret_cast<const float4*>(key_cache + kv_head_offset + t * head_dim);
        const float4* q_ptr_vec = reinterpret_cast<const float4*>(q_head);

        float score = 0.0f;
        for (int vd = 0; vd < vec_head_dim; ++vd) {
            Vector128 q_v, k_v;
            q_v.f4 = q_ptr_vec[vd];
            k_v.f4 = k_ptr_vec[vd];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                score += __half2float(q_v.h2[i].x) * __half2float(k_v.h2[i].x);
                score += __half2float(q_v.h2[i].y) * __half2float(k_v.h2[i].y);
            }
        }
        head_scores[t] = __float2half(score * scale);
    }
    __syncthreads();

    // 2. Softmax Max Reduction
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

    // 4. Weighted Sum over V (Vectorized Memory Access)
    float4* out_head_vec = reinterpret_cast<float4*>(att_out + h * head_dim);
    for (int vd = threadIdx.x; vd < vec_head_dim; vd += blockDim.x) {
        float acc[8] = {0.0f};

        for (int t = 0; t <= pos; ++t) {
            float score_t = __half2float(head_scores[t]);
            const float4* v_ptr_vec = reinterpret_cast<const float4*>(value_cache + kv_head_offset + t * head_dim);

            Vector128 v_val;
            v_val.f4 = v_ptr_vec[vd];

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i * 2 + 0] += score_t * __half2float(v_val.h2[i].x);
                acc[i * 2 + 1] += score_t * __half2float(v_val.h2[i].y);
            }
        }

        Vector128 out_v;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            out_v.h2[i].x = __float2half(acc[i * 2 + 0]);
            out_v.h2[i].y = __float2half(acc[i * 2 + 1]);
        }
        out_head_vec[vd] = out_v.f4;
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

// Standalone Parallel RoPE Kernel
__global__ void apply_rope_kernel(half* q, half* k, int n_heads, int n_kv_heads, int head_dim, int pos, float rope_theta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total_rotations = (n_heads + n_kv_heads) * half_hd;

    if (tid >= total_rotations) return;

    int total_q_rotations = n_heads * half_hd;
    bool is_q = (tid < total_q_rotations);

    int local_tid = is_q ? tid : (tid - total_q_rotations);
    int head_idx = local_tid / half_hd;
    int rot_i    = local_tid % half_hd;

    float freq = 1.0f / powf(rope_theta, (2.0f * rot_i) / static_cast<float>(head_dim));
    float val  = static_cast<float>(pos) * freq;
    float cos_v = cosf(val);
    float sin_v = sinf(val);

    half* target_buf = is_q ? q : k;
    int base_idx = head_idx * head_dim + rot_i;

    float v0 = __half2float(target_buf[base_idx]);
    float v1 = __half2float(target_buf[base_idx + half_hd]);

    target_buf[base_idx]           = __float2half(v0 * cos_v - v1 * sin_v);
    target_buf[base_idx + half_hd] = __float2half(v1 * cos_v + v0 * sin_v);
}

// Fused Attention Block Kernel (RMSNorm + Projection GEMV)
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
    half eps
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    extern __shared__ float s_mem_fp32[];
    float* s_reduce = s_mem_fp32;

    int vec_dim = dim / 8;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* norm_vec = reinterpret_cast<const float4*>(norm_weight);

    // 1. RMSNorm Reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        Vector128 xv;
        xv.f4 = x_vec[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f1 = __half2float(xv.h2[k].x);
            float f2 = __half2float(xv.h2[k].y);
            sum_sq += f1 * f1 + f2 * f2;
        }
    }
    s_reduce[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }

    float mean_sq = s_reduce[0] / static_cast<float>(dim);
    float inv_rms_f32 = rsqrtf(mean_sq + __half2float(eps));
    __syncthreads();

    // 2. Setup Target Row
    const float4* W_row_vec = nullptr;
    float bias_val = 0.0f;
    int target_idx = bid;
    bool is_q = (bid < q_dim);
    bool is_k = (bid >= q_dim && bid < q_dim + kv_dim);

    if (is_q) {
        W_row_vec = reinterpret_cast<const float4*>(W_q + target_idx * dim);
        bias_val = __half2float(b_q[target_idx]);
    } else if (is_k) {
        int k_idx = target_idx - q_dim;
        W_row_vec = reinterpret_cast<const float4*>(W_k + k_idx * dim);
        bias_val = __half2float(b_k[k_idx]);
    } else {
        int v_idx = target_idx - q_dim - kv_dim;
        W_row_vec = reinterpret_cast<const float4*>(W_v + v_idx * dim);
        bias_val = __half2float(b_v[v_idx]);
    }

    // 3. 128-bit Vectorized GEMV Dot-Product
    float thread_acc = 0.0f;
    for (int c = tid; c < vec_dim; c += blockDim.x) {
        Vector128 xv, nw, w;
        xv.f4 = x_vec[c];
        nw.f4 = norm_vec[c];
        w.f4  = W_row_vec[c];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float x0 = __half2float(xv.h2[k].x) * inv_rms_f32 * __half2float(nw.h2[k].x);
            float x1 = __half2float(xv.h2[k].y) * inv_rms_f32 * __half2float(nw.h2[k].y);
            thread_acc += __half2float(w.h2[k].x) * x0 + __half2float(w.h2[k].y) * x1;
        }
    }

    s_reduce[tid] = thread_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_reduce[tid] += s_reduce[tid + stride];
        __syncthreads();
    }

    if (tid == 0) {
        float out_val = s_reduce[0] + bias_val;
        if (is_q) {
            q_out[target_idx] = __float2half(out_val);
        } else if (is_k) {
            k_out[target_idx - q_dim] = __float2half(out_val);
        } else {
            v_out[target_idx - q_dim - kv_dim] = __float2half(out_val);
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
        q_out, k_out, v_out, dim, q_dim, kv_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());

    // Apply RoPE concurrently across Q and K outputs
    int total_rotations = (n_heads + n_kv_heads) * (head_dim / 2);
    int rope_blocks = (total_rotations + threads - 1) / threads;
    apply_rope_kernel<<<rope_blocks, threads, 0, stream>>>(
        q_out, k_out, n_heads, n_kv_heads, head_dim, pos, rope_theta
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused MLP Stage 1 Kernel (Dual-Reduced 128-bit Vectorized)
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
    float* s_reduce_gate = s_mlp_mem_fp32;
    float* s_reduce_up   = s_mlp_mem_fp32 + blockDim.x;

    int vec_dim = dim / 8;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* norm_vec = reinterpret_cast<const float4*>(norm_weight);

    // RMSNorm Reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        Vector128 xv;
        xv.f4 = x_vec[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f1 = __half2float(xv.h2[k].x);
            float f2 = __half2float(xv.h2[k].y);
            sum_sq += f1 * f1 + f2 * f2;
        }
    }
    s_reduce_gate[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_reduce_gate[tid] += s_reduce_gate[tid + stride];
        __syncthreads();
    }

    float mean_sq = s_reduce_gate[0] / static_cast<float>(dim);
    float inv_rms_f32 = rsqrtf(mean_sq + __half2float(eps));
    __syncthreads();

    // 128-bit Gate & Up GEMV
    const float4* gate_row_vec = reinterpret_cast<const float4*>(gate_weight + row * dim);
    const float4* up_row_vec   = reinterpret_cast<const float4*>(up_weight + row * dim);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (int c = tid; c < vec_dim; c += blockDim.x) {
        Vector128 xv, nw, gw, uw;
        xv.f4 = x_vec[c];
        nw.f4 = norm_vec[c];
        gw.f4 = gate_row_vec[c];
        uw.f4 = up_row_vec[c];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float x0 = __half2float(xv.h2[k].x) * inv_rms_f32 * __half2float(nw.h2[k].x);
            float x1 = __half2float(xv.h2[k].y) * inv_rms_f32 * __half2float(nw.h2[k].y);

            gate_acc += __half2float(gw.h2[k].x) * x0 + __half2float(gw.h2[k].y) * x1;
            up_acc   += __half2float(uw.h2[k].x) * x0 + __half2float(uw.h2[k].y) * x1;
        }
    }

    s_reduce_gate[tid] = gate_acc;
    s_reduce_up[tid]   = up_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_reduce_gate[tid] += s_reduce_gate[tid + stride];
            s_reduce_up[tid]   += s_reduce_up[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float gate_total = s_reduce_gate[0];
        float up_total   = s_reduce_up[0];
        float sigmoid_g  = 1.0f / (1.0f + expf(-gate_total));
        float swish      = gate_total * sigmoid_g;
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
    size_t shared_mem = 2 * threads * sizeof(float);

    fused_mlp_stage1_kernel<<<inter_dim, threads, shared_mem, stream>>>(
        x, norm_weight, gate_weight, up_weight, intermediate_out, dim, inter_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused MLP Stage 2
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
    int vec_inter = inter_dim / 8;

    const float4* down_row_vec = reinterpret_cast<const float4*>(down_weight + row * inter_dim);
    const float4* inter_in_vec = reinterpret_cast<const float4*>(intermediate_in);

    extern __shared__ float s_mlp_stage2_mem[];
    float thread_acc = 0.0f;

    for (int c = tid; c < vec_inter; c += blockDim.x) {
        Vector128 dw, in;
        dw.f4 = down_row_vec[c];
        in.f4 = inter_in_vec[c];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            thread_acc += __half2float(dw.h2[k].x) * __half2float(in.h2[k].x) +
                          __half2float(dw.h2[k].y) * __half2float(in.h2[k].y);
        }
    }

    s_mlp_stage2_mem[tid] = thread_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_mlp_stage2_mem[tid] += s_mlp_stage2_mem[tid + stride];
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

// Fused GEMV + Residual Addition Kernel
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
    int vec_cols = cols / 8;

    const float4* W_row_vec = reinterpret_cast<const float4*>(W + row * cols);
    const float4* in_vec    = reinterpret_cast<const float4*>(input);

    extern __shared__ float s_gemv_mem[];
    float thread_acc = 0.0f;

    for (int c = tid; c < vec_cols; c += blockDim.x) {
        Vector128 w, in;
        w.f4  = W_row_vec[c];
        in.f4 = in_vec[c];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            thread_acc += __half2float(w.h2[k].x) * __half2float(in.h2[k].x) +
                          __half2float(w.h2[k].y) * __half2float(in.h2[k].y);
        }
    }

    s_gemv_mem[tid] = thread_acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) s_gemv_mem[tid] += s_gemv_mem[tid + stride];
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

// RMSNorm
__global__ void rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ norm_weight,
    half* __restrict__ out,
    int dim,
    half eps
) {
    int tid = threadIdx.x;
    extern __shared__ float s_mem[];

    // Cast pointers to float4 for 128-bit vectorization
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* w_vec = reinterpret_cast<const float4*>(norm_weight);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int vec_dim = dim / 8;
    float sum_sq = 0.0f;

    // 1. Vectorized 128-bit FP32 Reduction (Coalesced)
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        Vector128 xv;
        xv.f4 = x_vec[i];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f1 = __half2float(xv.h2[k].x);
            float f2 = __half2float(xv.h2[k].y);
            sum_sq += f1 * f1 + f2 * f2;
        }
    }

    // Scalar tail handling
    for (int i = vec_dim * 8 + tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    s_mem[tid] = sum_sq;
    __syncthreads();

    // Standard Shared Memory Tree Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }

    float mean_sq = s_mem[0] / static_cast<float>(dim);
    float inv_rms_f32 = rsqrtf(mean_sq + __half2float(eps));
    __syncthreads();

    // 2. Vectorized 128-bit Write (Coalesced)
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        Vector128 xv, wv, out_v;
        xv.f4 = x_vec[i];
        wv.f4 = w_vec[i];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float o1 = __half2float(xv.h2[k].x) * inv_rms_f32 * __half2float(wv.h2[k].x);
            float o2 = __half2float(xv.h2[k].y) * inv_rms_f32 * __half2float(wv.h2[k].y);
            out_v.h2[k] = make_half2(__float2half(o1), __float2half(o2));
        }

        out_vec[i] = out_v.f4;
    }

    // Scalar tail write
    for (int i = vec_dim * 8 + tid; i < dim; i += blockDim.x) {
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

// Logits GEMV Kernel
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

    const float4* x_vec = reinterpret_cast<const float4*>(x_normed);
    const float4* w_vec = reinterpret_cast<const float4*>(lm_head_weight + static_cast<size_t>(v) * dim);

    int vec_dim = dim / 8;
    float dot_acc = 0.0f;

    // Vectorized 128-bit Dot Product (Coalesced reads across threads)
    for (int d = tid; d < vec_dim; d += blockDim.x) {
        Vector128 xv, wv;
        xv.f4 = x_vec[d];
        wv.f4 = w_vec[d];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            dot_acc += __half2float(xv.h2[k].x) * __half2float(wv.h2[k].x);
            dot_acc += __half2float(xv.h2[k].y) * __half2float(wv.h2[k].y);
        }
    }

    // Scalar tail handling
    for (int d = vec_dim * 8 + tid; d < dim; d += blockDim.x) {
        dot_acc += __half2float(x_normed[d]) * __half2float(lm_head_weight[static_cast<size_t>(v) * dim + d]);
    }

    s_logits_mem[tid] = dot_acc;
    __syncthreads();

    // Standard Shared Memory Tree Reduction
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

// GPU Repetition Penalty & Argmax
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