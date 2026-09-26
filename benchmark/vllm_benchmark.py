import os
import sys
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

VOCAB_SIZE   = 151936
MAX_CONTEXT  = 32000
WINDOW_LEN   = 500     # tokens decoded (and timed) per sampled depth
WARMUP_TOKENS = 500    # untimed warmup decode (heats CUDA graphs)

# Start position of each timed 0.5K decode window — identical to the CuQwen
# benchmark (cuqwen_benchmark.cu). Window i prefills `start` tokens then decodes
# WINDOW_LEN tokens, so decode runs at context depth start -> start+WINDOW_LEN.
WINDOW_STARTS = [500, 4500, 8500, 12500, 16500, 20500, 24500, 28500, 31500]


class BenchmarkInterval:
    def __init__(self, start_token, end_token, duration_ms, tok_per_sec):
        self.start_token = start_token
        self.end_token   = end_token
        self.duration_ms = duration_ms
        self.tok_per_sec = tok_per_sec


def resolve_model(model_size: str, quant: str) -> str:
    """FP16 -> HF model id; int8/int4 -> local compressed-tensors dir."""
    if quant == "fp16":
        return MODEL_MAP[model_size]
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(
        here, "..", "weights", "vllm_quant", f"{model_size.replace('.', '_')}_{quant}"
    ))
    if not os.path.isdir(path):
        sys.exit(
            f"[!] Quantized model not found: {path}\n"
            f"    Produce it first:\n"
            f"      python3 quantize_for_vllm.py --model={model_size} --quantization={quant}"
        )
    return path


async def decode_window(engine, dummy_tokens, start, req_id):
    """Prefill `start` tokens, then decode WINDOW_LEN tokens; time the decode.

    Prefill (parallel) reaches context depth `start`; only the streamed decode
    tokens are timed, so this measures steady-state decode throughput at that
    depth without paying to sequentially decode every preceding token.
    """
    prompt = dummy_tokens[:start] if start > 0 else [dummy_tokens[0]]
    sp = SamplingParams(temperature=0.0, max_tokens=WINDOW_LEN,
                        min_tokens=WINDOW_LEN, ignore_eos=True)

    prev_len = 0
    token_ts = []
    async for out in engine.generate({"prompt_token_ids": prompt}, sp, request_id=req_id):
        n = len(out.outputs[0].token_ids)
        # Record a timestamp whenever new decode token(s) arrive.
        for _ in range(n - prev_len):
            token_ts.append(time.perf_counter())
        prev_len = n

    # Exclude prefill: measure across the decode-token intervals only.
    if len(token_ts) < 2:
        return 0.0, 0.0
    elapsed_ms = (token_ts[-1] - token_ts[0]) * 1000.0
    steps = len(token_ts) - 1
    tok_sec = steps / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0.0
    return elapsed_ms, tok_sec


async def run_benchmark(model_size: str, quant: str):
    model_id = resolve_model(model_size, quant)
    precision = {"fp16": "FP16", "int8": "INT8 (weights-only, W8A16)",
                 "int4": "INT4 (weights-only, W4A16)"}[quant]

    print("========================================================================")
    print(f"   vLLM: Sampled Decode Benchmark Suite (Qwen2.5 {model_size.upper()})")
    print("========================================================================")
    print(f"\n[*] Precision : {precision}")
    print(f"[*] Loading weights from: {model_id}...")

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model                  = model_id,
        dtype                  = "float16",          # activations + KV cache stay FP16
        max_model_len          = MAX_CONTEXT + 64,
        max_num_seqs           = 1,
        gpu_memory_utilization = 0.88,               # headroom on 8GB cards
        disable_log_stats      = True,
        trust_remote_code      = True,
        enable_prefix_caching  = False,
    ))

    print("[✔] Weights & GPU State Initialized Successfully.\n")

    np.random.seed(42)
    dummy_tokens = np.random.randint(0, VOCAB_SIZE, size=MAX_CONTEXT).tolist()

    # --- Warmup (untimed): heats the decode CUDA graphs. ---
    print(f"[*] Warming up ({WARMUP_TOKENS} tokens)...")
    warmup_sp = SamplingParams(temperature=0.0, max_tokens=WARMUP_TOKENS,
                               min_tokens=WARMUP_TOKENS, ignore_eos=True)
    async for _ in engine.generate({"prompt_token_ids": [dummy_tokens[0]]},
                                    warmup_sp, request_id="warmup"):
        pass
    print("[✔] Warmup complete.\n")

    print("[*] Sampling decode throughput at 9 context depths (0.5K slice each)...")
    interval_results = []
    wall_start = time.perf_counter()

    for start in WINDOW_STARTS:
        end = start + WINDOW_LEN
        if end > MAX_CONTEXT:
            break
        elapsed_ms, tok_sec = await decode_window(engine, dummy_tokens, start, f"win_{start}")
        interval_results.append(BenchmarkInterval(start + 1, end, elapsed_ms, tok_sec))
        print(f"  [Sample] Context {end // 1000:2d}K"
              f" (tokens {start + 1:5d} - {end:5d})"
              f" | Speed: {tok_sec:8.2f} tok/s")

    total_wall_sec = time.perf_counter() - wall_start

    avg_tok_sec   = sum(r.tok_per_sec for r in interval_results) / len(interval_results)
    initial_speed = interval_results[0].tok_per_sec
    final_speed   = interval_results[-1].tok_per_sec
    decay_rate    = ((initial_speed - final_speed) / initial_speed) * 100.0 if initial_speed > 0 else 0.0
    measured_tokens = len(interval_results) * WINDOW_LEN

    print("\n========================================================================")
    print("                       vLLM BENCHMARK RESULTS                           ")
    print("========================================================================")
    print("| Token Range     | Duration (ms) | Speed (tokens/sec) | Context Slice |")
    print("+-----------------+---------------+--------------------+---------------+")
    for res in interval_results:
        print(f"| {res.start_token:6d} - {res.end_token:5d} | "
              f"{res.duration_ms:13.2f} | "
              f"{res.tok_per_sec:18.2f} | "
              f"{res.end_token // 1000:9d}k    |")
    print("+-----------------+---------------+--------------------+---------------+")
    print("\n========================================================================")
    print("                           OVERALL METRICS                              ")
    print("========================================================================")
    print(f"  • Model Architecture       : Qwen2.5 {model_size.upper()}")
    print(f"  • Weight Precision          : {precision}")
    print(f"  • Sampled Context Depths    : {len(interval_results)} windows (0.5K each)")
    print(f"  • Measured Tokens           : {measured_tokens} tokens (skips undecoded gaps)")
    print(f"  • Total Sampling Time       : {total_wall_sec:.2f} seconds")
    print(f"  • Average Speed             : {avg_tok_sec:.2f} tok/s")
    print(f"  • Initial Speed (~1K)       : {initial_speed:.2f} tok/s")
    print(f"  • Final Speed (~32K)        : {final_speed:.2f} tok/s")
    print(f"  • Speed Decay Rate          : {decay_rate:.2f} %")
    print("========================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM Sampled Decode Benchmark")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_MAP.keys()),
                        help="Model size variant: 0.5b, 1.5b, 3b, or 7b")
    parser.add_argument("--quantization", type=str, default="fp16", choices=["fp16", "int8", "int4"],
                        help="Weight precision: fp16 (default), int8 (W8A16) or int4 (W4A16). "
                             "int8/int4 load a checkpoint made by quantize_for_vllm.py.")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.model.lower(), args.quantization))
