import os
import time
import asyncio
import argparse
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}

TOTAL_TOKENS  = 32000
INTERVAL_STEP = 1000
WARMUP_TOKENS = 500
VOCAB_SIZE    = 151936

class BenchmarkInterval:
    def __init__(self, start_token: int, end_token: int, duration_ms: float, tok_per_sec: float):
        self.start_token = start_token
        self.end_token   = end_token
        self.duration_ms = duration_ms
        self.tok_per_sec = tok_per_sec

async def run_benchmark(model_id: str):
    print("========================================================================")
    print("         vLLM: Raw Decode CUDA-Equivalent Benchmark Suite (8000 Tokens) ")
    print("========================================================================")
    print(f"\n[*] Loading FP16 weights from: {model_id}...")

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model                 = model_id,
        dtype                 = "float16",
        max_model_len         = TOTAL_TOKENS + 64,
        max_num_seqs          = 1,
        disable_log_stats     = True,
        trust_remote_code     = True,
        enable_prefix_caching = False,
    ))

    print("[✔] Weights & GPU State Initialized Successfully.\n")

    np.random.seed(42)
    dummy_tokens = np.random.randint(0, VOCAB_SIZE, size=TOTAL_TOKENS).tolist()

    sp = SamplingParams(
        temperature = 0.0,
        max_tokens  = TOTAL_TOKENS,
        min_tokens  = TOTAL_TOKENS,
        ignore_eos  = True,
    )

    print(f"[*] Warming up ({WARMUP_TOKENS} tokens to heat decode CUDA graphs)...")
    warmup_sp = SamplingParams(
        temperature = 0.0,
        max_tokens  = WARMUP_TOKENS,
        min_tokens  = WARMUP_TOKENS,
        ignore_eos  = True,
    )
    async for _ in engine.generate(
        {"prompt_token_ids": [dummy_tokens[0]]}, warmup_sp, request_id="warmup"
    ):
        pass

    print("[✔] Warmup complete.\n")
    print("[*] Starting Raw Inference Benchmark\n")

    token_timestamps: list[float] = []
    wall_start = time.perf_counter()

    # Run 8000-token autoregressive generation
    async for _ in engine.generate(
        {"prompt_token_ids": [dummy_tokens[0]]}, sp, request_id="bench_run"
    ):
        token_timestamps.append(time.perf_counter())

    wall_end       = time.perf_counter()
    total_wall_sec = wall_end - wall_start

    interval_results: list[BenchmarkInterval] = []

    for slice_start in range(0, TOTAL_TOKENS, INTERVAL_STEP):
        slice_end = slice_start + INTERVAL_STEP

        t_start = wall_start if slice_start == 0 else token_timestamps[slice_start - 1]
        t_end   = token_timestamps[slice_end - 1]

        elapsed_ms = (t_end - t_start) * 1000.0
        tok_sec    = INTERVAL_STEP / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        interval_results.append(BenchmarkInterval(slice_start + 1, slice_end, elapsed_ms, tok_sec))
        print(f"  [Progress] Processed tokens {slice_start + 1:4d} to {slice_end:4d} "
              f"| Speed: {tok_sec:8.2f} tok/s")

    avg_tok_sec   = sum(r.tok_per_sec for r in interval_results) / len(interval_results)
    initial_speed = interval_results[0].tok_per_sec
    final_speed   = interval_results[-1].tok_per_sec
    decay_rate    = ((initial_speed - final_speed) / initial_speed) * 100.0 if initial_speed > 0 else 0.0

    print("\n========================================================================")
    print("                       vLLM BENCHMARK RESULTS                           ")
    print("========================================================================")
    print("| Token Range   | Duration (ms) | Speed (tokens/sec) | Context Slice   |")
    print("+---------------+---------------+--------------------+-----------------+")

    for res in interval_results:
        print(f"| {res.start_token:5d} - {res.end_token:4d} | "
              f"{res.duration_ms:13.2f} | "
              f"{res.tok_per_sec:18.2f} | "
              f"{res.end_token // 1000:8d}k context   |")

    print("+---------------+---------------+--------------------+-----------------+")
    print("\n========================================================================")
    print("                           OVERALL METRICS                              ")
    print("========================================================================")
    print(f"  • Total Generated Tokens   : {TOTAL_TOKENS} tokens")
    print(f"  • Total Time Elapsed       : {total_wall_sec:.2f} seconds")
    print(f"  • Average Speed            : {avg_tok_sec:.2f} tok/s")
    print(f"  • Initial Speed (0-1k)     : {initial_speed:.2f} tok/s")
    print(f"  • Final Speed (31k-32k)      : {final_speed:.2f} tok/s")
    print(f"  • Speed Decay Rate         : {decay_rate:.2f} %")
    print("========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Pure Decode Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Specify model size variant: 0.5b, 1.5b, 3b, or 7b",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(MODEL_MAP[args.model.lower()]))