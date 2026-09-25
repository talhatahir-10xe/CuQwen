#include "kernels.cuh"
#include <cmath>
#include <cuda_pipeline.h>
#include <mma.h>

union Vector128 {
    float4 f4;
    half2 h2[4];
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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

__global__ void gqa_attention_wmma_ksplit_stage1_kernel(
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
    using namespace nvcuda::wmma;
    const int total_len = *d_pos + 1;
    const int nparts    = gridDim.y;
    const int warp      = threadIdx.x >> 5;
    const int lane      = threadIdx.x & 31;
    const int part      = blockIdx.y;
    const int kv_head   = blockIdx.x;
    const int kv_group  = n_heads / n_kv_heads;
    const int ktiles_d  = head_dim / 16;
    const int ntiles_n  = head_dim / 16;

    extern __shared__ char kspl_smem[];
    half*  Qs      = reinterpret_cast<half*>(kspl_smem);
    half*  scores  = Qs + 16 * head_dim;
    half*  Ps      = scores + ATTN_KEYS_CAP * 16;
    float* scratch = reinterpret_cast<float*>(Ps + ATTN_KEYS_CAP * 16);
    float* sm      = scratch + KSPLIT_WARPS * 16 * 16;
    float* sl      = sm + 16;

    for (int i = threadIdx.x; i < 16 * head_dim; i += blockDim.x) {
        int h = i / head_dim, d = i % head_dim;
        Qs[i] = (h < kv_group) ? q[(size_t)(kv_head * kv_group + h) * head_dim + d] : __float2half(0.0f);
    }

    const int chunk = (total_len + nparts - 1) / nparts;
    const int t0    = part * chunk;
    const int t1    = min(t0 + chunk, total_len);
    if (t0 >= t1) { __syncthreads(); return; }
    const int nk    = t1 - t0;
    const int ntile = (nk + 15) / 16;

    const half* kbase = key_cache + (size_t)kv_head * max_seq_len * head_dim;
    const half* vbase = val_cache + (size_t)kv_head * max_seq_len * head_dim;

    float* wscratch = scratch + warp * 16 * 16;

    __syncthreads();

    for (int kt = warp; kt < ntile; kt += KSPLIT_WARPS) {
        fragment<accumulator, 16, 16, 16, float> c;
        fill_fragment(c, 0.0f);
        const half* krow = kbase + (size_t)(t0 + kt * 16) * head_dim;
        for (int dt = 0; dt < ktiles_d; ++dt) {
            fragment<matrix_a, 16, 16, 16, half, row_major> a;
            fragment<matrix_b, 16, 16, 16, half, col_major> b;
            load_matrix_sync(a, krow + dt * 16, head_dim);
            load_matrix_sync(b, Qs + dt * 16, head_dim);
            mma_sync(c, a, b, c);
        }
        store_matrix_sync(wscratch, c, 16, mem_row_major);
        __syncwarp();

        for (int i = lane; i < 16 * 16; i += 32) scores[kt * 16 * 16 + i] = __float2half(wscratch[i] * scale);
    }
    __syncthreads();

    if (warp == 0) {
        const int h   = lane & 7;          // head 0..7
        const int grp = lane >> 3;         // 0..3 (key stripe)
        float mx = -1e30f;
        for (int k = grp; k < nk; k += 4) mx = fmaxf(mx, __half2float(scores[k * 16 + h]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 8));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 16));
        float sum = 0.0f;
        for (int k = grp; k < nk; k += 4) {
            float e = __expf(__half2float(scores[k * 16 + h]) - mx);
            Ps[k * 16 + h] = __float2half(e);
            sum += e;
        }
        for (int k = nk + grp; k < ntile * 16; k += 4) Ps[k * 16 + h] = __float2half(0.0f);
        sum += __shfl_xor_sync(0xffffffff, sum, 8);
        sum += __shfl_xor_sync(0xffffffff, sum, 16);
        if (grp == 0) { sm[h] = mx; sl[h] = sum; }
    }
    __syncthreads();

    if (warp == 0 && lane < kv_group) {
        int slot = kv_head * PARTIAL_HEAD_SLOTS + lane;
        partial_max[slot * ATTN_PARTITIONS + part] = sm[lane];
        partial_sum[slot * ATTN_PARTITIONS + part] = sl[lane];
    }

    const int slot0 = kv_head * PARTIAL_HEAD_SLOTS;
    for (int nt = warp; nt < ntiles_n; nt += KSPLIT_WARPS) {
        fragment<accumulator, 16, 16, 16, float> o;
        fill_fragment(o, 0.0f);
        for (int kt = 0; kt < ntile; ++kt) {
            fragment<matrix_a, 16, 16, 16, half, col_major> pa;
            fragment<matrix_b, 16, 16, 16, half, row_major> vb;
            load_matrix_sync(pa, Ps + kt * 16 * 16, 16);
            load_matrix_sync(vb, vbase + (size_t)(t0 + kt * 16) * head_dim + nt * 16, head_dim);
            mma_sync(o, pa, vb, o);
        }
        store_matrix_sync(wscratch, o, 16, mem_row_major);
        __syncwarp();
        for (int i = lane; i < kv_group * 16; i += 32) {
            int h = i / 16, d = i % 16;
            partial_out[(size_t)((slot0 + h) * ATTN_PARTITIONS + part) * head_dim + nt * 16 + d] = wscratch[h * 16 + d];
        }
    }
}


__global__ void gqa_attention_flashdecode_stage2_kernel(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    half* __restrict__ attn_out,
    const int* __restrict__ d_pos,
    int max_seq_len,
    int head_dim,
    int num_partials,
    int n_kv_heads
) {
    (void)max_seq_len;
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bdx = blockDim.x;
    const int kv_group = gridDim.x / n_kv_heads;
    const int pslot = (h / kv_group) * PARTIAL_HEAD_SLOTS + (h % kv_group);

    const int total_len = *d_pos + 1;
    const int chunk     = (total_len + num_partials - 1) / num_partials;
    const int nonempty  = (total_len + chunk - 1) / chunk;

    extern __shared__ float stage2_smem[];
    float* s_max_arr = stage2_smem;
    float* s_sum_arr = stage2_smem + bdx;

    float t_max = -1e30f;
    for (int s = tid; s < nonempty; s += bdx)
        t_max = fmaxf(t_max, partial_max[pslot * ATTN_PARTITIONS + s]);
    s_max_arr[tid] = t_max;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_max_arr[tid] = fmaxf(s_max_arr[tid], s_max_arr[tid + stride]);
        __syncthreads();
    }
    const float global_max = s_max_arr[0];

    float t_sum = 0.0f;
    for (int s = tid; s < nonempty; s += bdx) {
        float sm = partial_max[pslot * ATTN_PARTITIONS + s];
        float ss = partial_sum[pslot * ATTN_PARTITIONS + s];
        t_sum += ss * __expf(sm - global_max);
    }
    s_sum_arr[tid] = t_sum;
    __syncthreads();
    for (int stride = bdx >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_sum_arr[tid] += s_sum_arr[tid + stride];
        __syncthreads();
    }
    const float inv_global_sum = 1.0f / (s_sum_arr[0] + 1e-9f);

    half* out_head = attn_out + h * head_dim;
    for (int d = tid; d < head_dim; d += bdx) {
        float final_acc = 0.0f;
        for (int s = 0; s < nonempty; ++s) {
            float rescale = __expf(partial_max[pslot * ATTN_PARTITIONS + s] - global_max);
            final_acc += partial_out[(size_t)(pslot * ATTN_PARTITIONS + s) * head_dim + d] * rescale;
        }
        out_head[d] = __float2half(final_acc * inv_global_sum);
    }
}

void launch_gqa_attention_decode(
    const half* q, const half* key_cache, const half* value_cache, half* attn_out,
    const int* d_pos, int n_heads, int n_kv_heads, int head_dim,
    int max_seq_len, float* d_partial_out, float* d_partial_max,
    float* d_partial_sum, cudaStream_t stream
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));


    dim3 grid_stage1(n_kv_heads, ATTN_PARTITIONS);
    int threads_stage1 = KSPLIT_WARPS * 32;
    size_t stage1_smem = (size_t)16 * head_dim * sizeof(half)
                       + (size_t)2 * ATTN_KEYS_CAP * 16 * sizeof(half)
                       + (size_t)KSPLIT_WARPS * 16 * 16 * sizeof(float)
                       + (size_t)32 * sizeof(float);
    static bool kspl_attr = false;
    if (!kspl_attr) {
        cudaFuncSetAttribute(gqa_attention_wmma_ksplit_stage1_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)stage1_smem);
        cudaFuncSetAttribute(gqa_attention_wmma_ksplit_stage1_kernel,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        kspl_attr = true;
    }
    gqa_attention_wmma_ksplit_stage1_kernel<<<grid_stage1, threads_stage1, stage1_smem, stream>>>(
        q, key_cache, value_cache,
        d_partial_out, d_partial_max, d_partial_sum,
        d_pos, n_heads, n_kv_heads, head_dim, max_seq_len, scale
    );

    size_t stage2_smem = 2 * 128 * sizeof(float);
    gqa_attention_flashdecode_stage2_kernel<<<n_heads, 128, stage2_smem, stream>>>(
        d_partial_out, d_partial_max, d_partial_sum,
        attn_out, d_pos, max_seq_len, head_dim, ATTN_PARTITIONS, n_kv_heads
    );
}

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

constexpr int QKV_WARPS_PER_BLOCK = GEMV_WARPS_PER_BLOCK;

__global__ void fused_attn_block_kernel(
    const half* __restrict__ xn,
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
    int kv_dim
) {
    const int lane = threadIdx.x & 31;
    const int out_idx = blockIdx.x * QKV_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    const int total = q_dim + 2 * kv_dim;
    if (out_idx >= total) return;

    const int vec_dim = dim / 8;
    const float4* x_g = reinterpret_cast<const float4*>(xn);

    const float4* W_row;
    float bias_val;
    const bool is_q = (out_idx < q_dim);
    const bool is_k = (out_idx >= q_dim && out_idx < q_dim + kv_dim);
    if (is_q) {
        W_row = reinterpret_cast<const float4*>(W_q + (size_t)out_idx * dim);
        bias_val = __half2float(b_q[out_idx]);
    } else if (is_k) {
        int ki = out_idx - q_dim;
        W_row = reinterpret_cast<const float4*>(W_k + (size_t)ki * dim);
        bias_val = __half2float(b_k[ki]);
    } else {
        int vi = out_idx - q_dim - kv_dim;
        W_row = reinterpret_cast<const float4*>(W_v + (size_t)vi * dim);
        bias_val = __half2float(b_v[vi]);
    }

    float acc = 0.0f;
    for (int c = lane; c < vec_dim; c += 32) {
        float4 xf = x_g[c];               // shared across warps -> L2 resident
        float4 wf = __ldg(W_row + c);
        const half2* xh = reinterpret_cast<const half2*>(&xf);
        const half2* wh = reinterpret_cast<const half2*>(&wf);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc = __fmaf_rn(__half2float(wh[k].x), __half2float(xh[k].x),
                  __fmaf_rn(__half2float(wh[k].y), __half2float(xh[k].y), acc));
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (lane == 0) {
        float out_val = acc + bias_val;
        if (is_q)      q_out[out_idx] = __float2half(out_val);
        else if (is_k) k_out[out_idx - q_dim] = __float2half(out_val);
        else           v_out[out_idx - q_dim - kv_dim] = __float2half(out_val);
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
    (void)norm_weight; (void)eps; // x is already RMS-normalized upstream
    int total = q_dim + (2 * kv_dim);
    int threads = QKV_WARPS_PER_BLOCK * 32;
    int blocks = (total + QKV_WARPS_PER_BLOCK - 1) / QKV_WARPS_PER_BLOCK;

    fused_attn_block_kernel<<<blocks, threads, 0, stream>>>(
        x, W_q, W_k, W_v, b_q, b_k, b_v,
        q_out, k_out, v_out, dim, q_dim, kv_dim
    );
    CUDA_CHECK(cudaGetLastError());

    int total_rotations = (n_heads + n_kv_heads) * (head_dim / 2);
    int rope_blocks = (total_rotations + threads - 1) / threads;
    apply_rope_kernel<<<rope_blocks, threads, 0, stream>>>(
        q_out, k_out, n_heads, n_kv_heads, head_dim, d_pos, rope_theta
    );
    CUDA_CHECK(cudaGetLastError());
}

constexpr int MLP1_WARPS_PER_BLOCK = GEMV_WARPS_PER_BLOCK;

__global__ void fused_mlp_stage1_kernel(
    const half* __restrict__ xn,
    const half* __restrict__ gate_weight,
    const half* __restrict__ up_weight,
    half* __restrict__ intermediate_out,
    int dim,
    int inter_dim
) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * MLP1_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= inter_dim) return;

    const int vec_dim = dim / 8;
    const float4* x_g    = reinterpret_cast<const float4*>(xn);
    const float4* gate_g = reinterpret_cast<const float4*>(gate_weight + (size_t)row * dim);
    const float4* up_g   = reinterpret_cast<const float4*>(up_weight   + (size_t)row * dim);

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (int c = lane; c < vec_dim; c += 32) {
        float4 xf = x_g[c];               // shared across warps -> L2 resident
        float4 gf = __ldg(gate_g + c);
        float4 uf = __ldg(up_g   + c);
        const half2* xh = reinterpret_cast<const half2*>(&xf);
        const half2* gh = reinterpret_cast<const half2*>(&gf);
        const half2* uh = reinterpret_cast<const half2*>(&uf);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float x0 = __half2float(xh[k].x);
            float x1 = __half2float(xh[k].y);
            gate_acc = __fmaf_rn(__half2float(gh[k].x), x0,
                       __fmaf_rn(__half2float(gh[k].y), x1, gate_acc));
            up_acc   = __fmaf_rn(__half2float(uh[k].x), x0,
                       __fmaf_rn(__half2float(uh[k].y), x1, up_acc));
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffff, gate_acc, off);
        up_acc   += __shfl_down_sync(0xffffffff, up_acc,   off);
    }

    if (lane == 0) {
        float sig = 1.0f / (1.0f + __expf(-gate_acc)); // SwiGLU: swish(gate) * up
        intermediate_out[row] = __float2half(gate_acc * sig * up_acc);
    }
}

void launch_fused_mlp_stage1(
    const half* x, const half* norm_weight,
    const half* gate_weight, const half* up_weight,
    half* intermediate_out, int dim, int inter_dim, half eps,
    cudaStream_t stream
) {
    (void)norm_weight; (void)eps; // x is already RMS-normalized upstream
    int threads = MLP1_WARPS_PER_BLOCK * 32;
    int blocks  = (inter_dim + MLP1_WARPS_PER_BLOCK - 1) / MLP1_WARPS_PER_BLOCK;
    fused_mlp_stage1_kernel<<<blocks, threads, 0, stream>>>(
        x, gate_weight, up_weight, intermediate_out, dim, inter_dim
    );
    CUDA_CHECK(cudaGetLastError());
}


constexpr int MLP2_WARPS_PER_BLOCK = GEMV_WARPS_PER_BLOCK;

__global__ void fused_mlp_stage2_kernel(
    half* __restrict__ x,
    const half* __restrict__ down_weight,
    const half* __restrict__ intermediate_in,
    int dim,
    int inter_dim
) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * MLP2_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= dim) return;

    const int vec_inter = inter_dim / 8;
    const float4* down_g = reinterpret_cast<const float4*>(down_weight + (size_t)row * inter_dim);
    const float4* inter_g = reinterpret_cast<const float4*>(intermediate_in);

    float acc = 0.0f;
    for (int c = lane; c < vec_inter; c += 32) {
        float4 dw = __ldg(down_g + c);
        float4 iv = inter_g[c];           // shared across warps -> L2 resident
        const half2* dh = reinterpret_cast<const half2*>(&dw);
        const half2* ih = reinterpret_cast<const half2*>(&iv);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc = __fmaf_rn(__half2float(dh[k].x), __half2float(ih[k].x),
                  __fmaf_rn(__half2float(dh[k].y), __half2float(ih[k].y), acc));
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (lane == 0)
        x[row] = __float2half(__half2float(x[row]) + acc);
}

void launch_fused_mlp_stage2(
    half* x, const half* down_weight, const half* intermediate_in,
    int dim, int inter_dim, cudaStream_t stream
) {
    int threads = MLP2_WARPS_PER_BLOCK * 32;
    int blocks  = (dim + MLP2_WARPS_PER_BLOCK - 1) / MLP2_WARPS_PER_BLOCK;
    fused_mlp_stage2_kernel<<<blocks, threads, 0, stream>>>(
        x, down_weight, intermediate_in, dim, inter_dim
    );
    CUDA_CHECK(cudaGetLastError());
}


constexpr int GEMV_ADD_WARPS_PER_BLOCK = GEMV_WARPS_PER_BLOCK;

__global__ void gemv_add_fp16_kernel(
    const half* __restrict__ W,
    const half* __restrict__ input,
    half* __restrict__ x_inout,
    int rows,
    int cols
) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * GEMV_ADD_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= rows) return;

    const int vec_cols = cols / 8;
    const float4* W_row_vec = reinterpret_cast<const float4*>(W + (size_t)row * cols);
    const float4* in_vec    = reinterpret_cast<const float4*>(input);

    float acc = 0.0f;
    for (int c = lane; c < vec_cols; c += 32) {
        float4 wf = __ldg(W_row_vec + c);
        float4 inf = in_vec[c];           // shared across warps -> L2 resident
        const half2* wh = reinterpret_cast<const half2*>(&wf);
        const half2* ih = reinterpret_cast<const half2*>(&inf);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc = __fmaf_rn(__half2float(wh[k].x), __half2float(ih[k].x),
                  __fmaf_rn(__half2float(wh[k].y), __half2float(ih[k].y), acc));
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (lane == 0)
        x_inout[row] = __float2half(__half2float(x_inout[row]) + acc);
}

void launch_gemv_add_fp16(
    const half* W,
    const half* input,
    half* x_inout,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int threads = GEMV_ADD_WARPS_PER_BLOCK * 32;
    int blocks  = (rows + GEMV_ADD_WARPS_PER_BLOCK - 1) / GEMV_ADD_WARPS_PER_BLOCK;
    gemv_add_fp16_kernel<<<blocks, threads, 0, stream>>>(W, input, x_inout, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

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

    for (int i = vec_dim * 8 + tid; i < dim; i += blockDim.x) {
        float val = __half2float(x[i]);
        sum_sq += val * val;
    }

    float block_sum_sq = block_reduce_sum(sum_sq, s_warp_mem);
    if (tid == 0) {
        float mean_sq = block_sum_sq / static_cast<float>(dim);
        s_warp_mem[0] = rsqrtf(mean_sq + __half2float(eps));
    }
    __syncthreads();
    float inv_rms_f32 = s_warp_mem[0];

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

constexpr int LOGITS_WARPS_PER_BLOCK = GEMV_WARPS_PER_BLOCK;

__global__ void compute_logits_kernel(
    const half* __restrict__ x_normed,
    const half* __restrict__ lm_head_weight,
    half* __restrict__ logits,
    int vocab_size,
    int dim
) {
    const int lane = threadIdx.x & 31;
    const int v    = blockIdx.x * LOGITS_WARPS_PER_BLOCK + (threadIdx.x >> 5);
    if (v >= vocab_size) return;

    const int vec_dim = dim / 8;
    const float4* x_g = reinterpret_cast<const float4*>(x_normed);
    const float4* w_g = reinterpret_cast<const float4*>(lm_head_weight + (size_t)v * dim);

    float acc = 0.0f;
    for (int d = lane; d < vec_dim; d += 32) {
        float4 xf = x_g[d];               // shared across warps -> L2 resident
        float4 wf = __ldg(w_g + d);
        const half2* xh = reinterpret_cast<const half2*>(&xf);
        const half2* wh = reinterpret_cast<const half2*>(&wf);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc = __fmaf_rn(__half2float(wh[k].x), __half2float(xh[k].x),
                  __fmaf_rn(__half2float(wh[k].y), __half2float(xh[k].y), acc));
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    if (lane == 0)
        logits[v] = __float2half(acc);
}

void launch_compute_logits(
    const half* x_normed, const half* lm_head_weight,
    half* logits, int vocab_size, int dim, cudaStream_t stream
) {
    int threads = LOGITS_WARPS_PER_BLOCK * 32;
    int blocks  = (vocab_size + LOGITS_WARPS_PER_BLOCK - 1) / LOGITS_WARPS_PER_BLOCK;
    compute_logits_kernel<<<blocks, threads, 0, stream>>>(
        x_normed, lm_head_weight, logits, vocab_size, dim
    );
    CUDA_CHECK(cudaGetLastError());
}

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