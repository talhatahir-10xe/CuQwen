import time
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}

TOTAL_TOKENS  = 8000
INTERVAL_STEP = 1000
WARMUP_TOKENS = 10

class BenchmarkInterval:
    def __init__(self, start_token: int, end_token: int, duration_ms: float, tok_per_sec: float):
        self.start_token = start_token
        self.end_token   = end_token
        self.duration_ms = duration_ms
        self.tok_per_sec = tok_per_sec

class TokenTimestampStreamer(TextStreamer):
    """Custom HuggingFace streamer that records accurate timestamps for each generated token."""
    def __init__(self, tokenizer):
        super().__init__(tokenizer, skip_prompt=True)
        self.timestamps = []

    def put(self, value):
        num_new_tokens = value.numel() if isinstance(value, torch.Tensor) else 1
        now = time.perf_counter()
        for _ in range(num_new_tokens):
            self.timestamps.append(now)

    def end(self):
        pass

def run_benchmark(model_id: str):
    print("========================================================================")
    print("       HuggingFace Transformers: Raw Decode Benchmark Suite (8000 Tokens)")
    print("========================================================================")
    print(f"\n[*] Loading FP16 weights from: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )

    print("[✔] Weights & CUDA Model Initialized Successfully.\n")

    prompt = "Write a comprehensive long-form essay about quantum computing and machine learning."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"[*] Warming up ({WARMUP_TOKENS} tokens to heat GPU state)...")
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=WARMUP_TOKENS,
            min_new_tokens=WARMUP_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    print("[✔] Warmup complete.\n")

    print("[*] Starting Raw Inference Benchmark\n")

    streamer = TokenTimestampStreamer(tokenizer)
    
    wall_start = time.perf_counter()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=TOTAL_TOKENS,
            min_new_tokens=TOTAL_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    total_wall_sec = wall_end - wall_start
    token_timestamps = streamer.timestamps[:TOTAL_TOKENS]

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
    print("                  HUGGINGFACE BENCHMARK RESULTS                         ")
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
    print(f"  • Final Speed (7k-8k)      : {final_speed:.2f} tok/s")
    print(f"  • Speed Decay Rate         : {decay_rate:.2f} %")
    print("========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Transformers Pure Decode Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_MAP.keys()),
        help="Specify model size variant: 0.5b, 1.5b, 3b, or 7b",
    )
    args = parser.parse_args()
    run_benchmark(MODEL_MAP[args.model.lower()])