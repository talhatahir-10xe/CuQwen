#include "kernels.cuh"
#include <cmath>

// Helper struct for 128-bit vectorized loading (8 x half = 16 bytes = 1 float4)
union Vector128 {
    float4 f4;
    half2 h2[4];
};

// Warp-Level Reduction Primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level reduction using Warp Shuffles and minimal Shared Memory (32 floats per block)
__device__ __forceinline__ float block_reduce_sum(float val, float* s_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads();

    // Reduce warp sums using the first warp
    val = (threadIdx.x < blockDim.x / 32) ? s_mem[lane] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* s_mem) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        s_mem[warp_id] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? s_mem[lane] : -1e9f;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

// Embedding Lookup Kernel
__global__ void embedding_lookup_kernel(const half* embed_table, const int* d_token_id, half* output, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_dim = dim / 8;

    int token_id = *d_token_id;
    const float4* embed_ptr = reinterpret_cast<const float4*>(embed_table + (size_t)token_id * dim);
    float4* out_ptr = reinterpret_cast<float4*>(output);

    if (idx < vec_dim) {
        out_ptr[idx] = embed_ptr[idx];
    }
}

void launch_embedding_lookup(const half* embed_table, const int* d_token_id, half* output, int dim, cudaStream_t stream) {
    int threads = 256;
    int blocks = ((dim / 8) + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(embed_table, d_token_id, output, dim);
    CUDA_CHECK(cudaGetLastError());
}

constexpr int SPLIT_K_SIZE = 256;

// FlashDecoding GQA Attention Stage 1 Kernel
__global__ void gqa_attention_flashdecode_stage1_kernel(
    const half* __restrict__ q, 
    const half* __restrict__ key_cache, 
    const half* __restrict__ val_cache,
    float* __restrict__ partial_out,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    const int* __restrict__ d_pos,
    int n_heads, 
    int n_kv_heads, 
    int head_dim,
    int max_seq_len, 
    float scale
) {
    int total_len = *d_pos + 1;
    const int h = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_head = h / kv_group;

    const int start_t = split_idx * SPLIT_K_SIZE;
    const int end_t = min(start_t + SPLIT_K_SIZE, total_len);

    if (start_t >= total_len) return;

    const int chunk_size = end_t - start_t;
    const half* q_head = q + h * head_dim;
    const int vec_dim = head_dim / 8;
    const float4* q_vec = reinterpret_cast<const float4*>(q_head);

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;

    // 1. Vectorized Q * K Dot Product
    for (int t_idx = tid; t_idx < chunk_size; t_idx += blockDim.x) {
        int t = start_t + t_idx;
        const float4* k_vec = reinterpret_cast<const float4*>(
            key_cache + (kv_head * max_seq_len + t) * head_dim
        );
        
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < vec_dim; ++i) {
            float4 qf4 = q_vec[i];
            float4 kf4 = k_vec[i];
            const half2* q_h2 = reinterpret_cast<const half2*>(&qf4);
            const half2* k_h2 = reinterpret_cast<const half2*>(&kf4);

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                half2 prod = __hmul2(q_h2[j], k_h2[j]);
                dot += __half2float(prod.x) + __half2float(prod.y);
            }
        }
        scores[t_idx] = dot * scale;
    }
    __syncthreads();

    // 2. Block Max Reduction
    float thread_max = -1e30f;
    for (int t_idx = tid; t_idx < chunk_size; t_idx += blockDim.x) {
        thread_max = fmaxf(thread_max, scores[t_idx]);
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }

    __shared__ float shared_red[4];
    const int lane = tid % 32;
    const int warp = tid / 32;
    if (lane == 0) shared_red[warp] = thread_max;
    __syncthreads();

    if (tid == 0) {
        float gmax = shared_red[0];
        for (int w = 1; w < (blockDim.x / 32); ++w) gmax = fmaxf(gmax, shared_red[w]);
        shared_red[0] = gmax;
    }
    __syncthreads();
    const float block_max = shared_red[0];

    // 3. Softmax Numerator & Sum Reduction
    float thread_sum = 0.0f;
    for (int t_idx = tid; t_idx < chunk_size; t_idx += blockDim.x) {
        float e = expf(scores[t_idx] - block_max);
        scores[t_idx] = e;
        thread_sum += e;
    }
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if (lane == 0) shared_red[warp] = thread_sum;
    __syncthreads();

    if (tid == 0) {
        float gsum = 0.0f;
        for (int w = 0; w < (blockDim.x / 32); ++w) gsum += shared_red[w];
        shared_red[0] = gsum;
    }
    __syncthreads();
    const float block_sum = shared_red[0];

    const int max_splits = (max_seq_len + SPLIT_K_SIZE - 1) / SPLIT_K_SIZE;
    const int split_out_idx = h * max_splits + split_idx;

    if (tid == 0) {
        partial_max[split_out_idx] = block_max;
        partial_sum[split_out_idx] = block_sum;
    }

    // 4. Coalesced Vectorized V-Cache Accumulation
    float* out_split_ptr = partial_out + split_out_idx * head_dim;
    const half* v_base = val_cache + (kv_head * max_seq_len + start_t) * head_dim;

    for (int d = tid * 2; d < head_dim; d += blockDim.x * 2) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        for (int t_idx = 0; t_idx < chunk_size; ++t_idx) {
            const half2* v_h2 = reinterpret_cast<const half2*>(&v_base[t_idx * head_dim + d]);
            half2 v_val = *v_h2;
            float score = scores[t_idx];
            acc0 += score * __half2float(v_val.x);
            acc1 += score * __half2float(v_val.y);
        }

        out_split_ptr[d]     = acc0;
        out_split_ptr[d + 1] = acc1;
    }
}

// FlashDecoding GQA Attention Stage 2 Kernel
__global__ void gqa_attention_flashdecode_stage2_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    half* __restrict__ attn_out,
    const int* __restrict__ d_pos,
    int max_seq_len,
    int head_dim
) {
    int total_len = *d_pos + 1;
    const int num_splits = (total_len + SPLIT_K_SIZE - 1) / SPLIT_K_SIZE;
    const int max_splits = (max_seq_len + SPLIT_K_SIZE - 1) / SPLIT_K_SIZE;
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdx = blockDim.x;

    extern __shared__ float stage2_smem[];
    float* s_max_arr = stage2_smem;
    float* s_sum_arr = stage2_smem + bdx;

    float t_max = -1e30f;
    for (int s = tid; s < num_splits; s += bdx) {
        t_max = fmaxf(t_max, partial_max[h * max_splits + s]);
    }
    s_max_arr[tid] = t_max;
    __syncthreads();

    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_max_arr[tid] = fmaxf(s_max_arr[tid], s_max_arr[tid + stride]);
        __syncthreads();
    }
    const float global_max = s_max_arr[0];

    float t_sum = 0.0f;
    for (int s = tid; s < num_splits; s += bdx) {
        float sm = partial_max[h * max_splits + s];
        float ss = partial_sum[h * max_splits + s];
        t_sum += ss * expf(sm - global_max);
    }
    s_sum_arr[tid] = t_sum;
    __syncthreads();

    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s_sum_arr[tid] += s_sum_arr[tid + stride];
        __syncthreads();
    }
    const float inv_global_sum = 1.0f / (s_sum_arr[0] + 1e-9f);

    half* out_head = attn_out + h * head_dim;
    for (int d = tid * 2; d < head_dim; d += bdx * 2) {
        float final_acc0 = 0.0f;
        float final_acc1 = 0.0f;

        #pragma unroll 4
        for (int s = 0; s < num_splits; ++s) {
            float rescale = expf(partial_max[h * max_splits + s] - global_max);
            int base_offset = (h * max_splits + s) * head_dim;
            final_acc0 += partial_out[base_offset + d]     * rescale;
            final_acc1 += partial_out[base_offset + d + 1] * rescale;
        }

        half2 result_h2 = __floats2half2_rn(
            final_acc0 * inv_global_sum,
            final_acc1 * inv_global_sum
        );
        *reinterpret_cast<half2*>(out_head + d) = result_h2;
    }
}

void launch_gqa_attention_decode(
    const half* q, const half* key_cache, const half* value_cache, half* attn_out,
    const int* d_pos, int n_heads, int n_kv_heads, int head_dim,
    int max_seq_len, float* d_partial_out, float* d_partial_max,
    float* d_partial_sum, cudaStream_t stream
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    int max_splits = (max_seq_len + SPLIT_K_SIZE - 1) / SPLIT_K_SIZE;

    dim3 grid_stage1(n_heads, max_splits);
    size_t shared_mem_stage1 = SPLIT_K_SIZE * sizeof(float);
    
    gqa_attention_flashdecode_stage1_kernel<<<grid_stage1, 128, shared_mem_stage1, stream>>>(
        q, key_cache, value_cache,
        d_partial_out, d_partial_max, d_partial_sum,
        d_pos, n_heads, n_kv_heads, head_dim, max_seq_len, scale
    );

    size_t stage2_smem = 2 * 128 * sizeof(float);
    gqa_attention_flashdecode_stage2_kernel<<<n_heads, 128, stage2_smem, stream>>>(
        d_partial_out, d_partial_max, d_partial_sum,
        attn_out, d_pos, max_seq_len, head_dim
    );
}

// KV Cache Store Kernel (Eliminates cudaMemcpyAsync calls in graph execution)
__global__ void kv_cache_store_kernel(
    const half* __restrict__ k_src,
    const half* __restrict__ v_src,
    half* __restrict__ layer_k_cache,
    half* __restrict__ layer_v_cache,
    const int* __restrict__ d_pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim
) {
    int pos = *d_pos;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_kv_heads * head_dim;
    if (tid >= total_elements) return;

    int h = tid / head_dim;
    int d = tid % head_dim;

    size_t dst_idx = (size_t)h * (size_t)max_seq_len * (size_t)head_dim + (size_t)pos * (size_t)head_dim + (size_t)d;
    layer_k_cache[dst_idx] = k_src[tid];
    layer_v_cache[dst_idx] = v_src[tid];
}

void launch_kv_cache_store(
    const half* k_src,
    const half* v_src,
    half* layer_k_cache,
    half* layer_v_cache,
    const int* d_pos,
    int n_kv_heads,
    int max_seq_len,
    int head_dim,
    cudaStream_t stream
) {
    int total_elements = n_kv_heads * head_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    kv_cache_store_kernel<<<blocks, threads, 0, stream>>>(
        k_src, v_src, layer_k_cache, layer_v_cache, d_pos, n_kv_heads, max_seq_len, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

// RoPE Kernel
__global__ void apply_rope_kernel(half* q, half* k, int n_heads, int n_kv_heads, int head_dim, const int* d_pos, float rope_theta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hd = head_dim / 2;
    int total_rotations = (n_heads + n_kv_heads) * half_hd;

    if (tid >= total_rotations) return;

    int pos = *d_pos;
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

// Fused Attention Block Kernel
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

    __shared__ float s_warp_mem[32];
    int vec_dim = dim / 8;
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* norm_vec = reinterpret_cast<const float4*>(norm_weight);

    // 1. RMSNorm Reduction via Warp Shuffle
    float sum_sq = 0.0f;
    #pragma unroll 4
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

    float block_sum_sq = block_reduce_sum(sum_sq, s_warp_mem);
    if (tid == 0) {
        float mean_sq = block_sum_sq / static_cast<float>(dim);
        s_warp_mem[0] = rsqrtf(mean_sq + __half2float(eps));
    }
    __syncthreads();
    float inv_rms_f32 = s_warp_mem[0];

    // 2. Target Row Setup
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
    #pragma unroll 4
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

    float total_acc = block_reduce_sum(thread_acc, s_warp_mem);

    if (tid == 0) {
        float out_val = total_acc + bias_val;
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
    const int* d_pos,
    float rope_theta,
    half eps,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = q_dim + (2 * kv_dim);

    fused_attn_block_kernel<<<blocks, threads, 0, stream>>>(
        x, norm_weight, W_q, W_k, W_v, b_q, b_k, b_v,
        q_out, k_out, v_out, dim, q_dim, kv_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());

    int total_rotations = (n_heads + n_kv_heads) * (head_dim / 2);
    int rope_blocks = (total_rotations + threads - 1) / threads;
    apply_rope_kernel<<<rope_blocks, threads, 0, stream>>>(
        q_out, k_out, n_heads, n_kv_heads, head_dim, d_pos, rope_theta
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused MLP Stage 1 Kernel
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
    const int tid  = threadIdx.x;
    const int row  = blockIdx.x;
    if (row >= inter_dim) return;

    __shared__ float s_red[32];

    const int vec_dim = dim / 8;
    const float4* x_g    = reinterpret_cast<const float4*>(x);
    const float4* nw_g   = reinterpret_cast<const float4*>(norm_weight);
    const float4* gate_g = reinterpret_cast<const float4*>(gate_weight + (size_t)row * dim);
    const float4* up_g   = reinterpret_cast<const float4*>(up_weight   + (size_t)row * dim);


    float sum_sq = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += blockDim.x) {
        float4 xf = __ldg(x_g + i);
        const half2* xh = reinterpret_cast<const half2*>(&xf);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float f0 = __half2float(xh[k].x);
            float f1 = __half2float(xh[k].y);
            sum_sq = __fmaf_rn(f0, f0, __fmaf_rn(f1, f1, sum_sq));
        }
    }

    // Warp-level reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, off);

    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;  // = 8 for blockDim=256

    if (lane == 0) s_red[warp_id] = sum_sq;
    __syncthreads();

    // Final reduce in warp 0
    if (warp_id == 0) {
        float v = (lane < n_warps) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0)
            s_red[0] = rsqrtf(__fmaf_rn(v, 1.0f / static_cast<float>(dim), __half2float(eps)));
    }
    __syncthreads();
    const float inv_rms = s_red[0];

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    #pragma unroll 4
    for (int c = tid; c < vec_dim; c += blockDim.x) {
        float4 xf  = __ldg(x_g  + c);
        float4 nwf = __ldg(nw_g + c);
        float4 gf  = __ldg(gate_g + c);
        float4 uf  = __ldg(up_g   + c);

        const half2* xh  = reinterpret_cast<const half2*>(&xf);
        const half2* nwh = reinterpret_cast<const half2*>(&nwf);
        const half2* gh  = reinterpret_cast<const half2*>(&gf);
        const half2* uh  = reinterpret_cast<const half2*>(&uf);

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            // xn = x * inv_rms * norm_weight (computed in registers)
            float xn0 = __half2float(xh[k].x) * inv_rms * __half2float(nwh[k].x);
            float xn1 = __half2float(xh[k].y) * inv_rms * __half2float(nwh[k].y);
            gate_acc = __fmaf_rn(__half2float(gh[k].x), xn0,
                       __fmaf_rn(__half2float(gh[k].y), xn1, gate_acc));
            up_acc   = __fmaf_rn(__half2float(uh[k].x), xn0,
                       __fmaf_rn(__half2float(uh[k].y), xn1, up_acc));
        }
    }

    // Single interleaved warp-level reduction (both accumulators, one shfl loop)
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffff, gate_acc, off);
        up_acc   += __shfl_down_sync(0xffffffff, up_acc,   off);
    }

    if (lane == 0) {
        s_red[warp_id]          = gate_acc;
        s_red[warp_id + n_warps] = up_acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float g = (lane < n_warps) ? s_red[lane]          : 0.0f;
        float u = (lane < n_warps) ? s_red[lane + n_warps] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffff, g, off);
            u += __shfl_down_sync(0xffffffff, u, off);
        }
        if (lane == 0) {
            // SwiGLU: swish(gate) * up
            float sig = 1.0f / (1.0f + __expf(-g));
            intermediate_out[row] = __float2half(g * sig * u);
        }
    }
}

void launch_fused_mlp_stage1(
    const half* x, const half* norm_weight,
    const half* gate_weight, const half* up_weight,
    half* intermediate_out, int dim, int inter_dim, half eps,
    cudaStream_t stream
) {
    fused_mlp_stage1_kernel<<<inter_dim, 256, 0, stream>>>(
        x, norm_weight, gate_weight, up_weight, intermediate_out, dim, inter_dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

// Fused MLP Stage 2
constexpr int MLP2_TILE_ROWS = 4;

__global__ void fused_mlp_stage2_kernel(
    half* __restrict__ x,
    const half* __restrict__ down_weight,
    const half* __restrict__ intermediate_in,
    int dim,
    int inter_dim
) {
    const int tid       = threadIdx.x;
    const int base_row  = blockIdx.x * MLP2_TILE_ROWS;
    const int vec_inter = inter_dim / 8;

    // 4 accumulators, one per tiled output row
    float acc[MLP2_TILE_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Precompute row pointers (4 rows). Guard rows that are out of range.
    const float4* down_rows[MLP2_TILE_ROWS];
    bool valid[MLP2_TILE_ROWS];
    #pragma unroll
    for (int r = 0; r < MLP2_TILE_ROWS; ++r) {
        int row = base_row + r;
        valid[r] = (row < dim);
        down_rows[r] = valid[r]
            ? reinterpret_cast<const float4*>(down_weight + (size_t)row * inter_dim)
            : nullptr;
    }

    const float4* inter_g = reinterpret_cast<const float4*>(intermediate_in);

    #pragma unroll 2
    for (int c = tid; c < vec_inter; c += blockDim.x) {
        // Load intermediate_in chunk — shared across all 4 rows of this block
        float4 iv = __ldg(inter_g + c);
        const half2* ih = reinterpret_cast<const half2*>(&iv);

        #pragma unroll
        for (int r = 0; r < MLP2_TILE_ROWS; ++r) {
            if (valid[r]) {
                float4 dw = __ldg(down_rows[r] + c);
                const half2* dh = reinterpret_cast<const half2*>(&dw);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    acc[r] = __fmaf_rn(__half2float(dh[k].x), __half2float(ih[k].x),
                             __fmaf_rn(__half2float(dh[k].y), __half2float(ih[k].y), acc[r]));
                }
            }
        }
    }

    // Warp reduce all 4 accumulators simultaneously
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < MLP2_TILE_ROWS; ++r)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
    }

    // Cross-warp reduction: use static smem (32*TILE_ROWS floats = 512 bytes)
    __shared__ float s_red[32 * MLP2_TILE_ROWS];

    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < MLP2_TILE_ROWS; ++r)
            s_red[r * 32 + warp_id] = acc[r];
    }
    __syncthreads();

    // Final reduce: warp 0 handles all rows
    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < MLP2_TILE_ROWS; ++r) {
            float v = (lane < n_warps) ? s_red[r * 32 + lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0 && valid[r]) {
                int row = base_row + r;
                x[row] = __float2half(__half2float(x[row]) + v);
            }
        }
    }
}

void launch_fused_mlp_stage2(
    half* x, const half* down_weight, const half* intermediate_in,
    int dim, int inter_dim, cudaStream_t stream
) {
    int blocks  = (dim + MLP2_TILE_ROWS - 1) / MLP2_TILE_ROWS;
    int threads = 256;
    fused_mlp_stage2_kernel<<<blocks, threads, 0, stream>>>(
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

    __shared__ float s_warp_mem[32];
    float thread_acc = 0.0f;

    #pragma unroll 4
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

    float total_acc = block_reduce_sum(thread_acc, s_warp_mem);

    if (tid == 0) {
        float orig_x = __half2float(x_inout[row]);
        x_inout[row] = __float2half(orig_x + total_acc);
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
    gemv_add_fp16_kernel<<<rows, threads, 0, stream>>>(W, input, x_inout, rows, cols);
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
    __shared__ float s_warp_mem[32];

    const float4* x_vec = reinterpret_cast<const float4*>(x);
    const float4* w_vec = reinterpret_cast<const float4*>(norm_weight);
    float4* out_vec = reinterpret_cast<float4*>(out);

    int vec_dim = dim / 8;
    float sum_sq = 0.0f;

    // 1. Vectorized 128-bit FP32 Reduction (Coalesced)
    #pragma unroll 4
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

    // Warp Shuffle Reduction
    float block_sum_sq = block_reduce_sum(sum_sq, s_warp_mem);
    if (tid == 0) {
        float mean_sq = block_sum_sq / static_cast<float>(dim);
        s_warp_mem[0] = rsqrtf(mean_sq + __half2float(eps));
    }
    __syncthreads();
    float inv_rms_f32 = s_warp_mem[0];

    // 2. Vectorized 128-bit Write (Coalesced)
    #pragma unroll 4
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

    rmsnorm_kernel<<<blocks, threads, 0, stream>>>(
        x, norm_weight, out, dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

// Logits GEMV Kernel
constexpr int LOGITS_TILE_ROWS = 4;

__global__ void compute_logits_kernel(
    const half* __restrict__ x_normed,
    const half* __restrict__ lm_head_weight,
    half* __restrict__ logits,
    int vocab_size,
    int dim
) {
    const int tid      = threadIdx.x;
    const int base_v   = blockIdx.x * LOGITS_TILE_ROWS;
    const int vec_dim  = dim / 8;

    float acc[LOGITS_TILE_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Precompute row pointers for the 4 vocabulary entries
    const float4* w_rows[LOGITS_TILE_ROWS];
    bool valid[LOGITS_TILE_ROWS];
    #pragma unroll
    for (int r = 0; r < LOGITS_TILE_ROWS; ++r) {
        int v = base_v + r;
        valid[r] = (v < vocab_size);
        w_rows[r] = valid[r]
            ? reinterpret_cast<const float4*>(lm_head_weight + (size_t)v * dim)
            : nullptr;
    }

    const float4* x_g = reinterpret_cast<const float4*>(x_normed);

    #pragma unroll 4
    for (int d = tid; d < vec_dim; d += blockDim.x) {
        float4 xf = __ldg(x_g + d);
        const half2* xh = reinterpret_cast<const half2*>(&xf);

        #pragma unroll
        for (int r = 0; r < LOGITS_TILE_ROWS; ++r) {
            if (valid[r]) {
                float4 wf = __ldg(w_rows[r] + d);
                const half2* wh = reinterpret_cast<const half2*>(&wf);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    acc[r] = __fmaf_rn(__half2float(wh[k].x), __half2float(xh[k].x),
                             __fmaf_rn(__half2float(wh[k].y), __half2float(xh[k].y), acc[r]));
                }
            }
        }
    }

    // Warp reduce all 4 accumulators in one interleaved loop
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        #pragma unroll
        for (int r = 0; r < LOGITS_TILE_ROWS; ++r)
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], off);
    }

    __shared__ float s_red[32 * LOGITS_TILE_ROWS];

    const int lane    = tid & 31;
    const int warp_id = tid >> 5;
    const int n_warps = blockDim.x >> 5;

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < LOGITS_TILE_ROWS; ++r)
            s_red[r * 32 + warp_id] = acc[r];
    }
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int r = 0; r < LOGITS_TILE_ROWS; ++r) {
            float v = (lane < n_warps) ? s_red[r * 32 + lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xffffffff, v, off);
            if (lane == 0 && valid[r]) {
                logits[base_v + r] = __float2half(v);
            }
        }
    }
}

void launch_compute_logits(
    const half* x_normed, const half* lm_head_weight,
    half* logits, int vocab_size, int dim, cudaStream_t stream
) {
    int blocks  = (vocab_size + LOGITS_TILE_ROWS - 1) / LOGITS_TILE_ROWS; // 37,984 blocks
    int threads = 256;
    compute_logits_kernel<<<blocks, threads, 0, stream>>>(
        x_normed, lm_head_weight, logits, vocab_size, dim
    );
    CUDA_CHECK(cudaGetLastError());
}

// GPU Repetition Penalty & Argmax Kernel
__global__ void repetition_penalty_kernel(
    half* logits,
    const int* d_history,
    const int* d_history_len,
    int vocab_size,
    half penalty
) {
    int history_len = *d_history_len;
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
    const int* d_history_len,
    int vocab_size,
    half penalty,
    cudaStream_t stream
) {
    int threads = 256;
    // Launch fixed 16 blocks (4096 tokens max history) for CUDA Graph static launch shape
    int blocks = 16;
    repetition_penalty_kernel<<<blocks, threads, 0, stream>>>(logits, d_history, d_history_len, vocab_size, penalty);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void argmax_kernel(const half* logits, int* out_token, int vocab_size) {
    int tid = threadIdx.x;
    float thread_max_val = -1e9f;
    int thread_max_idx = 0;

    #pragma unroll 4
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = __half2float(logits[i]);
        if (val > thread_max_val) {
            thread_max_val = val;
            thread_max_idx = i;
        }
    }

    // Warp-level reduction for Max value and index
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_down_sync(0xffffffff, thread_max_val, offset);
        int other_idx   = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
        if (other_val > thread_max_val) {
            thread_max_val = other_val;
            thread_max_idx = other_idx;
        }
    }

    __shared__ float s_max_vals[32];
    __shared__ int   s_max_indices[32];

    int lane = tid % 32;
    int warp_id = tid / 32;

    if (lane == 0) {
        s_max_vals[warp_id]    = thread_max_val;
        s_max_indices[warp_id] = thread_max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (tid < blockDim.x / 32) ? s_max_vals[lane] : -1e9f;
        int idx   = (tid < blockDim.x / 32) ? s_max_indices[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx   = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }

        if (tid == 0) {
            *out_token = idx;
        }
    }
}

void launch_argmax(const half* logits, int* d_sampled_token, int vocab_size, cudaStream_t stream) {
    int threads = 256;
    argmax_kernel<<<1, threads, 0, stream>>>(logits, d_sampled_token, vocab_size);
    CUDA_CHECK(cudaGetLastError());
}