#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "config.h"
#include "load_weights.h"
#include "model.cuh"

struct BenchmarkInterval {
    int start_token;
    int end_token;
    double duration_ms;
    double tok_per_sec;
};

int main() {
    std::cout << "===================================================================\n";
    std::cout << "        CUQWEN: CUDA Benchmark Suite (8000 Tokens)          \n";
    std::cout << "===================================================================\n\n";

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::string bin_path = "../weights/model_fp16_1_5b.bin";
    std::cout << "[*] Loading FP16 weights from: " << bin_path << "..." << std::endl;
    
    QwenWeights weights;
    QwenConfig config = load_weights(bin_path, weights);

    QwenState state;
    malloc_qwen_state(config, state);

    std::cout << "[✔] Weights & GPU State Initialized Successfully.\n" << std::endl;

    const int TOTAL_TOKENS = 8000;
    const int INTERVAL_STEP = 1000;
    
    // Generate 8000 pseudo-random token IDs within vocabulary range
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, config.vocab_size - 1);
    std::vector<int> dummy_tokens(TOTAL_TOKENS);
    for (int i = 0; i < TOTAL_TOKENS; ++i) {
        dummy_tokens[i] = dist(rng);
    }

    std::cout << "[*] Starting Raw Inference Benchmark" << std::endl;

    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));

    std::vector<BenchmarkInterval> interval_results;
    
    // Warmup forward pass
    qwen_forward(dummy_tokens[0], 0, config, state, weights, handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    int current_token = dummy_tokens[0];
    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int slice_start = 0; slice_start < TOTAL_TOKENS; slice_start += INTERVAL_STEP) {
        int slice_end = slice_start + INTERVAL_STEP;
        
        CUDA_CHECK(cudaEventRecord(start_event));

        for (int pos = slice_start; pos < slice_end; ++pos) {
            // Forward pass step
            current_token = qwen_forward(
                current_token,
                pos,
                config,
                state,
                weights,
                handle,
                dummy_tokens.data(),
                pos
            );
        }

        CUDA_CHECK(cudaEventRecord(end_event));
        CUDA_CHECK(cudaEventSynchronize(end_event));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));

        double tok_sec = (INTERVAL_STEP / (elapsed_ms / 1000.0));
        interval_results.push_back({slice_start + 1, slice_end, (double)elapsed_ms, tok_sec});
        
        std::cout << "  [Progress] Processed tokens " << std::setw(4) << slice_start + 1 
                  << " to " << std::setw(4) << slice_end 
                  << " | Speed: " << std::fixed << std::setprecision(2) << tok_sec << " tok/s" << std::endl;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

    double sum_tok_sec = 0.0;
    for (const auto& res : interval_results) {
        sum_tok_sec += res.tok_per_sec;
    }
    double avg_tok_sec = sum_tok_sec / interval_results.size();
    
    double initial_speed = interval_results.front().tok_per_sec;
    double final_speed = interval_results.back().tok_per_sec;
    double decay_rate = ((initial_speed - final_speed) / initial_speed) * 100.0;

    // Formatting output table
    std::cout << "\n========================================================================\n";
    std::cout << "                       CUDA BENCHMARK RESULTS                           \n";
    std::cout << "========================================================================\n";
    std::cout << "| Token Range   | Duration (ms) | Speed (tokens/sec) | Context Slice   |\n";
    std::cout << "+---------------+---------------+--------------------+-----------------+\n";

    for (const auto& res : interval_results) {
        std::cout << "| " << std::setw(5) << res.start_token << " - " << std::setw(4) << res.end_token << " | "
                  << std::setw(13) << std::fixed << std::setprecision(2) << res.duration_ms << " | "
                  << std::setw(18) << std::fixed << std::setprecision(2) << res.tok_per_sec << " | "
                  << std::setw(8) << res.end_token / 1000 << "k context   |\n";
    }

    std::cout << "+---------------+---------------+--------------------+-----------------+\n";
    std::cout << "\n========================================================================\n";
    std::cout << "                           OVERALL METRICS                              \n";
    std::cout << "========================================================================\n";
    std::cout << "  • Total Generated Tokens   : " << TOTAL_TOKENS << " tokens\n";
    std::cout << "  • Total Time Elapsed       : " << std::fixed << std::setprecision(2) << total_wall_sec << " seconds\n";
    std::cout << "  • Average Speed            : " << std::fixed << std::setprecision(2) << avg_tok_sec << " tok/s\n";
    std::cout << "  • Initial Speed (0-1k)     : " << std::fixed << std::setprecision(2) << initial_speed << " tok/s\n";
    std::cout << "  • Final Speed (7k-8k)      : " << std::fixed << std::setprecision(2) << final_speed << " tok/s\n";
    std::cout << "  • Speed Decay Rate         : " << std::fixed << std::setprecision(2) << decay_rate << " %\n";
    std::cout << "========================================================================\n\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));
    free_qwen_state(state);
    free_weights(weights, config);
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}