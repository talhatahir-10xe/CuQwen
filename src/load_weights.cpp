#include "load_weights.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>

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

#ifdef QUANT_ENABLED
// Load one quantized projection matrix: the packed/INT8 weight bytes followed
// by `rows*(in/QUANT_GROUP_SIZE)` FP16 group scales, matching the export layout.
// INT4 packs 2 weights per byte, so a row occupies in/2 bytes; INT8 uses in.
static void load_quant(FILE* f, qweight_t** weight, half** scale, size_t rows, size_t in) {
#ifdef QUANT_INT4
    const size_t weight_bytes = rows * (in / 2);
#else
    const size_t weight_bytes = rows * in;
#endif
    *weight = alloc_gpu<qweight_t>(f, weight_bytes);
    *scale  = alloc_gpu<half>(f, rows * (in / QUANT_GROUP_SIZE));
}
#endif

QwenConfig load_weights(const std::string& bin_path, QwenWeights& weights) {
    FILE* f = fopen(bin_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[!] Error: Cannot open binary file at: " << bin_path << std::endl;
        exit(EXIT_FAILURE);
    }

    int32_t header[10] = {0};
    if (fread(header, sizeof(int32_t), 10, f) != 10) {
        std::cerr << "[!] Error: Failed to read binary weight header!" << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (header[0] != QwenConfig::magic) {
        std::cerr << "[!] Error: Magic number mismatch! Expected 0x" << std::hex << QwenConfig::magic
                  << ", found 0x" << header[0] << std::dec << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    QwenConfig config;

    if (header[1] != config.vocab_size || header[2] != config.dim ||
        header[3] != config.intermediate_size || header[4] != config.n_layers ||
        header[5] != config.n_heads || header[6] != config.n_kv_heads ||
        header[7] != config.head_dim) {
        std::cerr << "[!] Error: Binary header dimensions do not match compiled QwenConfig parameters!" << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    if (header[9] != config.quant_type) {
        auto quant_name = [](int q) {
            return q == 2 ? "int4" : q == 1 ? "int8" : q == 0 ? "fp16" : "unknown";
        };
        const char* want = quant_name(config.quant_type);
        const char* got  = quant_name(header[9]);
        std::cerr << "[!] Error: Quantization mismatch! This binary was compiled for '" << want
                  << "' weights but the file '" << bin_path << "' is '" << got << "'.\n"
                  << "    Re-export with '--quantization=" << want
                  << "' or rebuild with '-Dquant=" << got << "'." << std::endl;
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fseek(f, 256, SEEK_SET);

    const size_t q_dim  = (size_t)config.n_heads    * config.head_dim;
    const size_t kv_dim = (size_t)config.n_kv_heads * config.head_dim;
    const size_t dim    = config.dim;
    const size_t inter  = config.intermediate_size;

    std::cout << "[*] Allocating and transferring weights to GPU VRAM...\n" << std::endl;

    std::cout << "  [+] Loading embed_tokens.weight..." << std::flush;
    weights.embed_tokens = alloc_gpu<half>(f, (size_t)config.vocab_size * dim);
    std::cout << " Done." << std::endl;

    weights.layers.resize(config.n_layers);
    for (int l = 0; l < config.n_layers; ++l) {
        std::cout << "  [+] Loading Transformer Layer " << std::setw(2) << l + 1 
                  << "/" << config.n_layers << "...\r" << std::flush;

        LayerWeights& lw = weights.layers[l];

        lw.input_layernorm_weight          = alloc_gpu<half>(f, dim);
#ifdef QUANT_ENABLED
        load_quant(f, &lw.q_proj_weight, &lw.q_proj_scale, q_dim, dim);
        lw.q_proj_bias                     = alloc_gpu<half>(f, q_dim);
        load_quant(f, &lw.k_proj_weight, &lw.k_proj_scale, kv_dim, dim);
        lw.k_proj_bias                     = alloc_gpu<half>(f, kv_dim);
        load_quant(f, &lw.v_proj_weight, &lw.v_proj_scale, kv_dim, dim);
        lw.v_proj_bias                     = alloc_gpu<half>(f, kv_dim);
        load_quant(f, &lw.o_proj_weight, &lw.o_proj_scale, dim, q_dim);
        lw.post_attention_layernorm_weight = alloc_gpu<half>(f, dim);
        load_quant(f, &lw.gate_proj_weight, &lw.gate_proj_scale, inter, dim);
        load_quant(f, &lw.up_proj_weight,   &lw.up_proj_scale,   inter, dim);
        load_quant(f, &lw.down_proj_weight, &lw.down_proj_scale, dim, inter);
#else
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
#endif
    }
    std::cout << "  [+] Loaded all " << config.n_layers << " Transformer Layers successfully.            " << std::endl;

    std::cout << "  [+] Loading final model.norm.weight..." << std::flush;
    weights.norm_weight = alloc_gpu<half>(f, dim);
    std::cout << " Done." << std::endl;

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
#ifdef QUANT_ENABLED
        CUDA_CHECK(cudaFree(lw.q_proj_scale));
        CUDA_CHECK(cudaFree(lw.k_proj_scale));
        CUDA_CHECK(cudaFree(lw.v_proj_scale));
        CUDA_CHECK(cudaFree(lw.o_proj_scale));
        CUDA_CHECK(cudaFree(lw.gate_proj_scale));
        CUDA_CHECK(cudaFree(lw.up_proj_scale));
        CUDA_CHECK(cudaFree(lw.down_proj_scale));
#endif
    }
    CUDA_CHECK(cudaFree(weights.norm_weight));

    if (weights.lm_head_weight != nullptr && weights.lm_head_weight != weights.embed_tokens) {
        CUDA_CHECK(cudaFree(weights.lm_head_weight));
    }
}