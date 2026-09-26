#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "config.h"
#include "load_weights.h"
#include "model.cuh"

struct BenchmarkInterval {
    int start_token;
    int end_token;
    double duration_ms;
    double tok_per_sec;
};

// Weight precision this binary was compiled for (see -Dquant / config.h).
static const char* precision_label() {
#if defined(QUANT_INT8)
    return "INT8 (weights-only, W8A16)";
#elif defined(QUANT_INT4)
    return "INT4 (weights-only, W4A16)";
#else
    return "FP16";
#endif
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "   CUQWEN: Bare-Metal CUDA Benchmark Suite (Qwen2.5 " << QwenConfig::model_name << ")\n";
    std::cout << "========================================================================\n\n";

    std::string bin_path = "../" + std::string(QwenConfig::bin_path);
    std::cout << "[*] Precision: " << precision_label() << "\n";
    std::cout << "[*] Loading weights from: " << bin_path << "..." << std::endl;

    QwenWeights weights;
    QwenConfig config = load_weights(bin_path, weights);

    QwenState state;
    malloc_qwen_state(config, state);

    std::cout << "[✔] Weights & GPU State Initialized Successfully.\n" << std::endl;

    // -----------------------------------------------------------------------
    // Sampled benchmarking strategy
    //
    // Rather than decoding the full 32K window (slow), we sample throughput at
    // 9 context depths. Each sample times a 500-token (0.5K) decode slice ending
    // on a round context boundary: 0.5-1K, 4.5-5K, 8.5-9K, ... 28.5-29K, 31.5-32K.
    // The first 0-0.5K slice is an (untimed) warmup that also triggers the
    // one-time CUDA-graph capture.
    //
    // The decode work *between* sampled windows is skipped: we simply advance
    // `pos` to the next window. The KV cache is preallocated and the attention
    // kernel does work proportional to (pos + 1) regardless of whether the
    // skipped cache slots hold real or zeroed data, so per-token throughput at
    // each sampled depth stays representative while the run finishes far faster.
    // (Tokens are random, so outputs are meaningless — this measures speed only.)
    // -----------------------------------------------------------------------
    const int MAX_CONTEXT = std::min(32000, QwenConfig::max_seq_len - 2);
    const int WARMUP_TOKENS = 500;   // 0 -> 0.5K, untimed
    const int WINDOW_LEN    = 500;   // length of each timed slice

    // Start position of each timed 0.5K window (window i measures [start, start+500)).
    const int window_starts[] = {500, 4500, 8500, 12500, 16500, 20500, 24500, 28500, 31500};

    // Pseudo-random token IDs (indexed by position; content irrelevant to timing).
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, config.vocab_size - 1);
    std::vector<int> dummy_tokens(MAX_CONTEXT);
    for (int i = 0; i < MAX_CONTEXT; ++i) {
        dummy_tokens[i] = dist(rng);
    }

    std::cout << "[*] Sampling decode throughput at 9 context depths (0.5K slice each)..." << std::endl;

    cudaEvent_t start_event, end_event;
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&end_event));

    std::vector<BenchmarkInterval> interval_results;

    // --- Warmup (0 -> 0.5K): first call captures the CUDA graph; untimed. ---
    int current_token = dummy_tokens[0];
    for (int pos = 0; pos < WARMUP_TOKENS; ++pos) {
        current_token = qwen_forward(
            dummy_tokens[pos], pos, config, state, weights, dummy_tokens.data(), pos
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto wall_start = std::chrono::high_resolution_clock::now();

    // --- Timed sampled windows ---
    for (int start : window_starts) {
        int end = start + WINDOW_LEN;
        if (end > MAX_CONTEXT) break;   // stay within the model's context capacity

        CUDA_CHECK(cudaEventRecord(start_event));

        for (int pos = start; pos < end; ++pos) {
            current_token = qwen_forward(
                dummy_tokens[pos], pos, config, state, weights, dummy_tokens.data(), pos
            );
        }

        CUDA_CHECK(cudaEventRecord(end_event));
        CUDA_CHECK(cudaEventSynchronize(end_event));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));

        double tok_sec = WINDOW_LEN / (elapsed_ms / 1000.0);
        interval_results.push_back({start + 1, end, (double)elapsed_ms, tok_sec});

        std::cout << "  [Sample] Context " << std::setw(2) << end / 1000 << "K"
                  << " (tokens " << std::setw(5) << start + 1 << " - " << std::setw(5) << end << ")"
                  << " | Speed: " << std::fixed << std::setprecision(2) << tok_sec << " tok/s" << std::endl;
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

    // Stats calculations
    double sum_tok_sec = 0.0;
    for (const auto& res : interval_results) {
        sum_tok_sec += res.tok_per_sec;
    }
    double avg_tok_sec = interval_results.empty() ? 0.0 : sum_tok_sec / interval_results.size();

    double initial_speed = interval_results.empty() ? 0.0 : interval_results.front().tok_per_sec;
    double final_speed = interval_results.empty() ? 0.0 : interval_results.back().tok_per_sec;
    double decay_rate = (initial_speed > 0.0) ? (((initial_speed - final_speed) / initial_speed) * 100.0) : 0.0;

    int measured_tokens = (int)interval_results.size() * WINDOW_LEN;

    // Formatting output table
    std::cout << "\n========================================================================\n";
    std::cout << "                       CUDA BENCHMARK RESULTS                           \n";
    std::cout << "========================================================================\n";
    std::cout << "| Token Range     | Duration (ms) | Speed (tokens/sec) | Context Slice |\n";
    std::cout << "+-----------------+---------------+--------------------+---------------+\n";

    for (const auto& res : interval_results) {
        std::cout << "| " << std::setw(6) << res.start_token << " - " << std::setw(5) << res.end_token << " | "
                  << std::setw(13) << std::fixed << std::setprecision(2) << res.duration_ms << " | "
                  << std::setw(18) << std::fixed << std::setprecision(2) << res.tok_per_sec << " | "
                  << std::setw(9) << res.end_token / 1000 << "k    |\n";
    }

    std::cout << "+-----------------+---------------+--------------------+---------------+\n";
    std::cout << "\n========================================================================\n";
    std::cout << "                           OVERALL METRICS                              \n";
    std::cout << "========================================================================\n";
    std::cout << "  • Model Architecture       : Qwen2.5 " << QwenConfig::model_name << "\n";
    std::cout << "  • Weight Precision          : " << precision_label() << "\n";
    std::cout << "  • Sampled Context Depths    : " << interval_results.size() << " windows (0.5K each)\n";
    std::cout << "  • Measured Tokens           : " << measured_tokens << " tokens (skips undecoded gaps)\n";
    std::cout << "  • Total Sampling Time       : " << std::fixed << std::setprecision(2) << total_wall_sec << " seconds\n";
    std::cout << "  • Average Speed             : " << std::fixed << std::setprecision(2) << avg_tok_sec << " tok/s\n";
    std::cout << "  • Initial Speed (~1K)       : " << std::fixed << std::setprecision(2) << initial_speed << " tok/s\n";
    std::cout << "  • Final Speed (~32K)        : " << std::fixed << std::setprecision(2) << final_speed << " tok/s\n";
    std::cout << "  • Speed Decay Rate          : " << std::fixed << std::setprecision(2) << decay_rate << " %\n";
    std::cout << "========================================================================\n\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(end_event));
    free_qwen_state(state);
    free_weights(weights, config);

    return 0;
}
