# =============================================================================
# ollama_benchmark.py — Ollama Pure Decode Benchmark (8000 Tokens, FP16)
# =============================================================================

import time
import asyncio
import argparse
from ollama import AsyncClient

MODEL_MAP = {
    "0.5b": "qwen2.5:0.5b-fp16",
    "1.5b": "qwen2.5:1.5b-fp16",
    "3b":   "qwen2.5:3b-fp16",
    "7b":   "qwen2.5:7b-fp16",
}

TOTAL_TOKENS  = 32000
INTERVAL_STEP = 1000
WARMUP_TOKENS = 500

class BenchmarkInterval:
    def __init__(self, start_token: int, end_token: int, duration_ms: float, tok_per_sec: float):
        self.start_token = start_token
        self.end_token   = end_token
        self.duration_ms = duration_ms
        self.tok_per_sec = tok_per_sec

async def run_benchmark(model_id: str):
    print("========================================================================")
    print("           Ollama API: Pure Decode Benchmark (FP16 Weights & KV)        ")
    print("========================================================================")
    print(f"\n[*] Target Model Tag  : {model_id}")
    print(f"[*] Precision Setup   : FP16 Weights | FP16 KV Cache (f16_kv=True)")

    client = AsyncClient()

    # Enforce FP16 KV cache alongside full offloading
    options = {
        "num_gpu": -1,                 # Offload all layers to GPU
        "num_ctx": TOTAL_TOKENS + 512, # Ensure sufficient KV cache space
        "temperature": 0.0,            # Greedy decoding
        "num_thread": 8,               # CPU worker threads
        "f16_kv": True,                # Force FP16 KV Cache precision
    }

    prompt = "Count continuously from 1 to 100000, writing each number out with detailed descriptions: 1, 2, 3,"

    # Warmup
    print(f"[*] Warming up ({WARMUP_TOKENS} tokens to heat GPU VRAM state)...")
    stream = await client.generate(
        model=model_id,
        prompt="Hello",
        stream=True,
        options={**options, "num_predict": WARMUP_TOKENS},
    )
    async for _ in stream:
        pass
    print("[✔] Warmup complete.\n")

    print("[*] Starting Raw Inference Benchmark\n")

    chunk_timestamps: list[float] = []
    wall_start = time.perf_counter()

    tokens_generated_so_far = 0
    total_eval_duration_ns = 0

    while tokens_generated_so_far < TOTAL_TOKENS:
        tokens_remaining = TOTAL_TOKENS - tokens_generated_so_far

        stream = await client.generate(
            model=model_id,
            prompt=prompt,
            stream=True,
            options={**options, "num_predict": tokens_remaining},
        )

        sub_count = 0
        async for chunk in stream:
            now = time.perf_counter()
            chunk_timestamps.append(now)

            is_done = chunk.get("done", False) if isinstance(chunk, dict) else getattr(chunk, "done", False)
            eval_count = (chunk.get("eval_count", 0) if isinstance(chunk, dict) else getattr(chunk, "eval_count", 0)) or 0
            eval_duration = (chunk.get("eval_duration", 0) if isinstance(chunk, dict) else getattr(chunk, "eval_duration", 0)) or 0

            if is_done and eval_count > 0:
                sub_count = eval_count
                total_eval_duration_ns += eval_duration

        if sub_count == 0:
            sub_count = tokens_remaining

        tokens_generated_so_far += sub_count
        prompt += " continuation..."

    wall_end = time.perf_counter()
    total_wall_sec = wall_end - wall_start

    actual_tokens = tokens_generated_so_far
    num_chunks = len(chunk_timestamps)

    interval_results: list[BenchmarkInterval] = []

    for slice_start_tok in range(0, actual_tokens, INTERVAL_STEP):
        slice_end_tok = min(slice_start_tok + INTERVAL_STEP, actual_tokens)
        tokens_in_slice = slice_end_tok - slice_start_tok

        start_idx = int((slice_start_tok / actual_tokens) * num_chunks)
        end_idx   = int((slice_end_tok / actual_tokens) * num_chunks) - 1
        end_idx   = max(start_idx, min(end_idx, num_chunks - 1))

        t_start = wall_start if start_idx == 0 else chunk_timestamps[start_idx - 1]
        t_end   = chunk_timestamps[end_idx]

        elapsed_ms = (t_end - t_start) * 1000.0
        tok_sec    = tokens_in_slice / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0

        interval_results.append(BenchmarkInterval(slice_start_tok + 1, slice_end_tok, elapsed_ms, tok_sec))
        print(f"  [Progress] Processed tokens {slice_start_tok + 1:4d} to {slice_end_tok:4d} "
              f"| Speed: {tok_sec:8.2f} tok/s")

    avg_tok_sec   = sum(r.tok_per_sec for r in interval_results) / len(interval_results) if interval_results else 0.0
    initial_speed = interval_results[0].tok_per_sec if interval_results else 0.0
    final_speed   = interval_results[-1].tok_per_sec if interval_results else 0.0
    decay_rate    = ((initial_speed - final_speed) / initial_speed) * 100.0 if initial_speed > 0 else 0.0

    print("\n========================================================================")
    print("                   OLLAMA BENCHMARK RESULTS (FP16)                      ")
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
    print(f"  • Model & Precision        : {model_id} (FP16 Weights + FP16 KV)")
    print(f"  • Total Generated Tokens   : {actual_tokens} tokens")
    print(f"  • Total Time Elapsed       : {total_wall_sec:.2f} seconds")
    if total_eval_duration_ns > 0:
        server_tok_sec = actual_tokens / (total_eval_duration_ns / 1e9)
        print(f"  • Server Pure Eval Speed   : {server_tok_sec:.2f} tok/s (Ollama Internal Metric)")
    print(f"  • Average Speed (Streaming): {avg_tok_sec:.2f} tok/s")
    print(f"  • Initial Speed (0-1k)     : {initial_speed:.2f} tok/s")
    print(f"  • Final Speed (31k-32k)      : {final_speed:.2f} tok/s")
    print(f"  • Speed Decay Rate         : {decay_rate:.2f} %")
    print("========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Pure Decode Benchmark (FP16)")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Specify model size variant: 0.5b, 1.5b, 3b, or 7b",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(MODEL_MAP[args.model.lower()]))