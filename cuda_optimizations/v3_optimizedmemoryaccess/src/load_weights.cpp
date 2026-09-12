#include "load_weights.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>

// Generic GPU Memory Allocation Helper from Binary Stream
template <typename T>
static T* alloc_gpu(FILE* f, size_t n) {
    std::vector<T> buf(n);
    if (fread(buf.data(), sizeof(T), n, f) != n) {
        std::cerr << "\n[!] Error: Unexpected end of file while reading tensor data!" << std::endl;
        exit(EXIT_FAILURE);
    }
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(ptr, buf.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

QwenConfig load_weights(const std::string& bin_path, QwenWeights& weights) {
    FILE* f = fopen(bin_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[!] Error: Cannot open binary file at: " << bin_path << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read header (9 x int32)
    int32_t header[9] = {0};
    if (fread(header, sizeof(int32_t), 9, f) != 9) {
        std::cerr << "[!] Error: Failed to read binary weight header!" << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // Magic number verification
    if (header[0] != QwenConfig::magic) {
        std::cerr << "[!] Error: Magic number mismatch! Expected 0x" << std::hex << QwenConfig::magic
                  << ", found 0x" << header[0] << std::dec << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    QwenConfig config;

    // Validate binary metadata against compiled header parameters
    if (header[1] != config.vocab_size || header[2] != config.dim ||
        header[3] != config.intermediate_size || header[4] != config.n_layers ||
        header[5] != config.n_heads || header[6] != config.n_kv_heads ||
        header[7] != config.head_dim) {
        std::cerr << "[!] Error: Binary header dimensions do not match compiled QwenConfig parameters!" << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // Jump to byte offset 256 (end of header pad)
    fseek(f, 256, SEEK_SET);

    const size_t q_dim  = (size_t)config.n_heads    * config.head_dim;
    const size_t kv_dim = (size_t)config.n_kv_heads * config.head_dim;
    const size_t dim    = config.dim;
    const size_t inter  = config.intermediate_size;

    std::cout << "[*] Allocating and transferring weights to GPU VRAM...\n" << std::endl;

    // 1. Load Token Embeddings
    std::cout << "  [+] Loading embed_tokens.weight..." << std::flush;
    weights.embed_tokens = alloc_gpu<half>(f, (size_t)config.vocab_size * dim);
    std::cout << " Done." << std::endl;

    // 2. Load Transformer Block Layers
    weights.layers.resize(config.n_layers);
    for (int l = 0; l < config.n_layers; ++l) {
        std::cout << "  [+] Loading Transformer Layer " << std::setw(2) << l + 1 
                  << "/" << config.n_layers << "...\r" << std::flush;

        LayerWeights& lw = weights.layers[l];

        lw.input_layernorm_weight          = alloc_gpu<half>(f, dim);
        lw.q_proj_weight                   = alloc_gpu<half>(f, q_dim * dim);
        lw.q_proj_bias                     = alloc_gpu<half>(f, q_dim);
        lw.k_proj_weight                   = alloc_gpu<half>(f, kv_dim * dim);
        lw.k_proj_bias                     = alloc_gpu<half>(f, kv_dim);
        lw.v_proj_weight                   = alloc_gpu<half>(f, kv_dim * dim);
        lw.v_proj_bias                     = alloc_gpu<half>(f, kv_dim);
        lw.o_proj_weight                   = alloc_gpu<half>(f, dim * q_dim);
        lw.post_attention_layernorm_weight = alloc_gpu<half>(f, dim);
        lw.gate_proj_weight                = alloc_gpu<half>(f, inter * dim);
        lw.up_proj_weight                  = alloc_gpu<half>(f, inter * dim);
        lw.down_proj_weight                = alloc_gpu<half>(f, dim * inter);
    }
    std::cout << "  [+] Loaded all 28 Transformer Layers successfully.            " << std::endl;

    // 3. Load Final LayerNorm
    std::cout << "  [+] Loading final model.norm.weight..." << std::flush;
    weights.norm_weight = alloc_gpu<half>(f, dim);
    std::cout << " Done." << std::endl;

    // 4. Handle Tied LM Head Embedding
    long cur = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, cur, SEEK_SET);

    if ((size_t)(end - cur) >= (size_t)config.vocab_size * dim * sizeof(half)) {
        std::cout << "  [+] Loading separate lm_head.weight..." << std::flush;
        weights.lm_head_weight = alloc_gpu<half>(f, (size_t)config.vocab_size * dim);
        std::cout << " Done." << std::endl;
    } else {
        std::cout << "  [+] lm_head is tied to embed_tokens (reusing pointer)." << std::endl;
        weights.lm_head_weight = weights.embed_tokens;
    }

    fclose(f);
    return config;
}

void free_weights(QwenWeights& weights, const QwenConfig& config) {
    CUDA_CHECK(cudaFree(weights.embed_tokens));
    for (int l = 0; l < config.n_layers; ++l) {
        LayerWeights& lw = weights.layers[l];
        CUDA_CHECK(cudaFree(lw.input_layernorm_weight));
        CUDA_CHECK(cudaFree(lw.q_proj_weight));
        CUDA_CHECK(cudaFree(lw.q_proj_bias));
        CUDA_CHECK(cudaFree(lw.k_proj_weight));
        CUDA_CHECK(cudaFree(lw.k_proj_bias));
        CUDA_CHECK(cudaFree(lw.v_proj_weight));
        CUDA_CHECK(cudaFree(lw.v_proj_bias));
        CUDA_CHECK(cudaFree(lw.o_proj_weight));
        CUDA_CHECK(cudaFree(lw.post_attention_layernorm_weight));
        CUDA_CHECK(cudaFree(lw.gate_proj_weight));
        CUDA_CHECK(cudaFree(lw.up_proj_weight));
        CUDA_CHECK(cudaFree(lw.down_proj_weight));
    }
    CUDA_CHECK(cudaFree(weights.norm_weight));

    if (weights.lm_head_weight != nullptr && weights.lm_head_weight != weights.embed_tokens) {
        CUDA_CHECK(cudaFree(weights.lm_head_weight));
    }
}