// Standalone per-kernel micro-profiler using CUDA events (no perf counters).
// Times each decode kernel in isolation at a fixed context depth so we can see
// exactly where decode time goes. Uses dummy device buffers sized from QwenConfig
// (kernel timing is independent of actual weight values).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <iomanip>
#include <cstdio>
#include "config.h"
#include "kernels.cuh"

static int POS = 16000;

template <typename T> T* dmalloc(size_t n) { T* p; cudaMalloc(&p, n * sizeof(T)); cudaMemset(p, 0, n*sizeof(T)); return p; }

#define TIME(label, niter, call) do { \
    for (int i=0;i<5;i++){ call; } cudaDeviceSynchronize(); \
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); \
    cudaEventRecord(a); \
    for (int i=0;i<(niter);i++){ call; } \
    cudaEventRecord(b); cudaEventSynchronize(b); \
    float ms=0; cudaEventElapsedTime(&ms,a,b); \
    double us = ms*1000.0/(niter); \
    printf("  %-26s %8.2f us\n", label, us); \
    total_layer_us += us; \
    cudaEventDestroy(a); cudaEventDestroy(b); \
} while(0)

int main(int argc, char** argv){
    if (argc>1) POS = atoi(argv[1]);
    QwenConfig cfg;
    const int dim=cfg.dim, inter=cfg.intermediate_size, head_dim=cfg.head_dim;
    const int n_heads=cfg.n_heads, n_kv=cfg.n_kv_heads;
    const int q_dim=n_heads*head_dim, kv_dim=n_kv*head_dim;
    const int msl=cfg.max_seq_len, vocab=cfg.vocab_size;
    printf("Model %s  POS=%d (context %dK)\n", cfg.model_name, POS, (POS+1)/1000);

    // buffers
    half *x=dmalloc<half>(dim), *xb=dmalloc<half>(dim), *att=dmalloc<half>(dim);
    half *q=dmalloc<half>(q_dim), *k=dmalloc<half>(kv_dim), *v=dmalloc<half>(kv_dim);
    half *gate=dmalloc<half>(inter), *up=dmalloc<half>(inter);
    half *logits=dmalloc<half>(vocab);
    half *norm_w=dmalloc<half>(dim);
    int *d_pos=dmalloc<int>(1); cudaMemcpy(d_pos,&POS,sizeof(int),cudaMemcpyHostToDevice);
    int *d_tok=dmalloc<int>(1), *d_samp=dmalloc<int>(1), *d_hist=dmalloc<int>(msl), *d_hlen=dmalloc<int>(1);
    int hlen=500; cudaMemcpy(d_hlen,&hlen,sizeof(int),cudaMemcpyHostToDevice);

    // quant weights (qweight_t). Row stride handles int4 packing.
    auto qm=[&](size_t rows,size_t cols){ return dmalloc<qweight_t>(rows*WEIGHT_ROW_STRIDE(cols)); };
    auto sm=[&](size_t rows,size_t cols){ return dmalloc<half>(rows*(cols/QUANT_GROUP_SIZE)); };
    qweight_t *Wq=qm(q_dim,dim),*Wk=qm(kv_dim,dim),*Wv=qm(kv_dim,dim),*Wo=qm(dim,q_dim);
    qweight_t *Wg=qm(inter,dim),*Wu=qm(inter,dim),*Wd=qm(dim,inter);
    half *Sq=0,*Sk=0,*Sv=0,*So=0,*Sg=0,*Su=0,*Sd=0;
#ifdef QUANT_ENABLED
    Sq=sm(q_dim,dim);Sk=sm(kv_dim,dim);Sv=sm(kv_dim,dim);So=sm(dim,q_dim);
    Sg=sm(inter,dim);Su=sm(inter,dim);Sd=sm(dim,inter);
#endif
    half *bq=dmalloc<half>(q_dim),*bk=dmalloc<half>(kv_dim),*bv=dmalloc<half>(kv_dim);
    half *embed=dmalloc<half>((size_t)vocab*dim);
    half *lmhead=dmalloc<half>((size_t)vocab*dim);
    // KV cache for one layer
    half *kc=dmalloc<half>((size_t)n_kv*msl*head_dim), *vc=dmalloc<half>((size_t)n_kv*msl*head_dim);
    size_t splits=(size_t)n_kv*PARTIAL_HEAD_SLOTS*ATTN_PARTITIONS;
    float *po=dmalloc<float>(splits*head_dim), *pmx=dmalloc<float>(splits), *psm=dmalloc<float>(splits);

    half eps=__float2half(cfg.norm_eps);
    cudaStream_t s=0;
    double total_layer_us=0;
    int N=300;
    printf("--- per-layer kernels (x%d layers) ---\n", cfg.n_layers);
    TIME("rmsnorm", N, launch_rmsnorm(x,norm_w,att,dim,eps,s));
    TIME("fused_attn_block(qkv+rope)", N, launch_fused_attn_block(att,norm_w,Wq,Wk,Wv,Sq,Sk,Sv,bq,bk,bv,q,k,v,dim,q_dim,kv_dim,n_heads,n_kv,head_dim,d_pos,cfg.rope_theta,eps,s));
    TIME("kv_cache_store", N, launch_kv_cache_store(k,v,kc,vc,d_pos,n_kv,msl,head_dim,s));
    TIME("gqa_attention_decode", N, launch_gqa_attention_decode(q,kc,vc,xb,d_pos,n_heads,n_kv,head_dim,msl,po,pmx,psm,s));
    TIME("o_proj gemv_add", N, launch_gemv_add_fp16(Wo,So,xb,x,dim,q_dim,s));
    TIME("mlp_stage1(gate+up)", N, launch_fused_mlp_stage1(att,norm_w,Wg,Wu,Sg,Su,gate,dim,inter,eps,s));
    TIME("mlp_stage2(down)", N, launch_fused_mlp_stage2(x,Wd,Sd,gate,dim,inter,s));
    double per_layer=total_layer_us;
    printf("  layer subtotal:            %8.2f us  => x%d = %.2f us\n", per_layer, cfg.n_layers, per_layer*cfg.n_layers);

    double total_layer_us2=total_layer_us; total_layer_us=0;
    printf("--- once per token ---\n");
    TIME("embedding_lookup", N, launch_embedding_lookup(embed,d_tok,x,dim,s));
    TIME("final rmsnorm", N, launch_rmsnorm(x,norm_w,xb,dim,eps,s));
    TIME("compute_logits(lm_head)", N, launch_compute_logits(xb,lmhead,logits,vocab,dim,s));
    TIME("repetition_penalty", N, launch_apply_repetition_penalty(logits,d_hist,d_hlen,vocab,__float2half(1.1f),s));
    TIME("argmax", N, launch_argmax(logits,d_samp,vocab,s));
    double once=total_layer_us;
    double est_token = per_layer*cfg.n_layers + once;
    printf("--- totals ---\n");
    printf("  layers total: %.1f us | once: %.1f us | est/token: %.1f us => %.1f tok/s\n",
        per_layer*cfg.n_layers, once, est_token, 1e6/est_token);
    return 0;
}
