# =============================================================================
# ollama_benchmark.py — Ollama sampled decode benchmark (FP16 / INT8 / INT4)
#
# Mirrors cuqwen_benchmark.cu / vllm_benchmark.py: instead of decoding the full
# 32K window, it samples decode throughput at 9 context depths (a 0.5K decode
# slice ending at 1/5/9/.../29/32K). For each depth it prefills a prompt of
# ~`start` tokens, then decodes ~500 tokens; Ollama's server-reported
# `eval_duration` (pure decode time, prefill excluded) is used for tok/s — the
# same decode-only measurement as the other two engines.
#
# Quantization (see --quantization): FP16, or Ollama's Q8_0 (int8) / Q4_0
# (int4) GGUFs. These are symmetric, no zero-point, weights-only, with an FP16
# KV cache — matching CuQwen/vLLM EXCEPT for two unavoidable GGUF deviations:
#   (1) block size 32, not 128 (GGUF has no group-128 symmetric type), and
#   (2) token embeddings + lm_head are also quantized (the pre-built GGUFs
#       quantize them; only a custom llama-quantize run keeps them FP16).
# Norms stay F32 in GGUF, as in the other engines.
#
# NOTE on decode length: Ollama exposes no `ignore_eos`, so the model may emit
# EOS before `num_predict` — decode length per window varies (small quantized
# models often stop after a few dozen tokens). This does NOT bias the result:
# the reported tok/s is Ollama's server-measured pure per-token decode rate
# (eval_count / eval_duration), which is stable across token counts (verified:
# a 40-token window and a 500-token window at the same depth report the same
# tok/s). The counting-sequence prompt is used to reach the target depth and to
# maximize decode length where the model cooperates.
# =============================================================================

import time
import argparse
from ollama import Client

# transformers is only used to build prompts of a precise token length so the
# sampled context depths line up with the vLLM/CuQwen benchmarks.
from transformers import AutoTokenizer

TOKENIZER_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MODEL_SIZES = ["0.5b", "1.5b", "3b", "7b"]
QUANT_SUFFIX = {"fp16": "fp16", "int8": "q8_0", "int4": "q4_0"}

MAX_CONTEXT   = 32000
WINDOW_LEN    = 500     # tokens decoded (and timed) per sampled depth
WINDOW_STARTS = [500, 4500, 8500, 12500, 16500, 20500, 24500, 28500, 31500]

# Ollama offers no `ignore_eos`, so to reliably decode ~WINDOW_LEN tokens we make
# the PROMPT ITSELF a coherent counting sequence ("1, 2, 3, ..., N,") of the
# desired token length. The model is mid-count, so it simply keeps emitting the
# next integers for the whole window without an early EOS, while the KV length
# equals the intended context depth. (An arbitrary/random filler makes small
# quantized models emit EOS almost immediately — hence the counting context.)


class BenchmarkInterval:
    def __init__(self, depth, decode_tokens, duration_ms, tok_per_sec):
        self.depth = depth
        self.decode_tokens = decode_tokens
        self.duration_ms = duration_ms
        self.tok_per_sec = tok_per_sec


def model_tag(size: str, quant: str) -> str:
    return f"qwen2.5:{size}-{QUANT_SUFFIX[quant]}"


def build_counting_corpus(tokenizer, min_tokens: int):
    """Token IDs of '1, 2, 3, ...' long enough to slice any target depth from.

    Starts from a token-count estimate (~2.5 tokens/number) and grows only if
    needed, so the encoded corpus stays modestly above `min_tokens` rather than
    ballooning past the tokenizer's max-length warning threshold.
    """
    n = max(1000, min_tokens // 3)
    while True:
        text = ", ".join(str(i) for i in range(1, n)) + ", "
        ids = tokenizer.encode(text)
        if len(ids) >= min_tokens + 32:
            return ids
        n = int(n * 1.5)


def build_prompt(tokenizer, corpus_ids, target_tokens: int) -> str:
    """A counting prompt of exactly ~target_tokens, ending mid-sequence."""
    return tokenizer.decode(corpus_ids[:target_tokens])


def run_benchmark(size: str, quant: str):
    tag = model_tag(size, quant)
    precision = {"fp16": "FP16", "int8": "INT8 (Q8_0, block-32, no zero-point)",
                 "int4": "INT4 (Q4_0, block-32, no zero-point)"}[quant]

    print("========================================================================")
    print(f"        Ollama: Sampled Decode Benchmark (Qwen2.5 {size.upper()})")
    print("========================================================================")
    print(f"[*] Model tag : {tag}")
    print(f"[*] Precision : {precision} | FP16 KV cache")

    client = Client()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    corpus_ids = build_counting_corpus(tokenizer, MAX_CONTEXT)

    # Constant across all calls so Ollama does not reload between depths.
    base_options = {
        "num_gpu": -1,            # offload all layers to GPU
        "num_ctx": 32768,         # KV cache large enough for the deepest window
        "temperature": 0.0,       # greedy
        "f16_kv": True,           # FP16 KV cache
    }

    # --- Warmup (untimed): load model into VRAM AND heat the decode path.
    # Two passes so the decode kernels/graphs are hot before we time anything
    # (a single short prefill would only warm the prefill path). The counting
    # context makes the model actually decode ~WINDOW_LEN tokens here. ---
    print("[*] Warming up (prefill + decode path)...")
    warm_prompt = build_prompt(tokenizer, corpus_ids, 128)
    for _ in range(2):
        wr = client.generate(model=tag, prompt=warm_prompt,
                             options={**base_options, "num_predict": WINDOW_LEN}, stream=False)
    print(f"[✔] Warmup complete (decoded {int(wr.get('eval_count', 0))} tokens).\n")

    print("[*] Sampling decode throughput at 9 context depths (0.5K slice each)...")
    results = []
    wall_start = time.perf_counter()

    for start in WINDOW_STARTS:
        if start + WINDOW_LEN > MAX_CONTEXT:
            break
        prompt = build_prompt(tokenizer, corpus_ids, start)
        resp = client.generate(model=tag, prompt=prompt,
                               options={**base_options, "num_predict": WINDOW_LEN}, stream=False)

        # Ollama server metrics: eval_* is the decode phase (prefill excluded).
        depth      = int(resp.get("prompt_eval_count", start))
        dec_tokens = int(resp.get("eval_count", 0))
        eval_ns    = int(resp.get("eval_duration", 0))
        tok_sec    = dec_tokens / (eval_ns / 1e9) if eval_ns > 0 else 0.0
        dur_ms     = eval_ns / 1e6

        results.append(BenchmarkInterval(depth, dec_tokens, dur_ms, tok_sec))
        # tok/s is Ollama's server-side per-token decode rate (eval_count /
        # eval_duration), which is stable regardless of how many tokens were
        # produced. We only warn on a truly tiny sample or a capped depth.
        warn = ""
        if depth < 0.8 * start:
            warn += f"  [!] depth {depth} << target {start}: Ollama capped num_ctx (insufficient VRAM)"
        if dec_tokens < 16:
            warn += f"  [!] only {dec_tokens} decode tokens (rate still valid, but thin sample)"
        print(f"  [Sample] Context ~{round(depth/1000):2d}K"
              f" (prompt {depth:5d} tok, decoded {dec_tokens:4d})"
              f" | Speed: {tok_sec:8.2f} tok/s{warn}")

    total_wall_sec = time.perf_counter() - wall_start

    avg_tok_sec   = sum(r.tok_per_sec for r in results) / len(results) if results else 0.0
    initial_speed = results[0].tok_per_sec if results else 0.0
    final_speed   = results[-1].tok_per_sec if results else 0.0
    decay_rate    = ((initial_speed - final_speed) / initial_speed) * 100.0 if initial_speed > 0 else 0.0

    print("\n========================================================================")
    print("                     OLLAMA BENCHMARK RESULTS                           ")
    print("========================================================================")
    print("| Context Depth | Decoded | Decode (ms) | Speed (tokens/sec) |")
    print("+---------------+---------+-------------+--------------------+")
    for r in results:
        print(f"| {r.depth:8d} tok | {r.decode_tokens:7d} | "
              f"{r.duration_ms:11.2f} | {r.tok_per_sec:18.2f} |")
    print("+---------------+---------+-------------+--------------------+")
    print("\n========================================================================")
    print("                           OVERALL METRICS                              ")
    print("========================================================================")
    print(f"  • Model & Precision        : {tag}  ({precision})")
    print(f"  • Sampled Context Depths    : {len(results)} windows (~0.5K decode each)")
    print(f"  • Total Sampling Time       : {total_wall_sec:.2f} seconds")
    print(f"  • Average Speed             : {avg_tok_sec:.2f} tok/s")
    print(f"  • Initial Speed (~1K)       : {initial_speed:.2f} tok/s")
    print(f"  • Final Speed (~32K)        : {final_speed:.2f} tok/s")
    print(f"  • Speed Decay Rate          : {decay_rate:.2f} %")
    print("========================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ollama Sampled Decode Benchmark (FP16/INT8/INT4)")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_SIZES,
                        help="Model size variant: 0.5b, 1.5b, 3b, or 7b")
    parser.add_argument("--quantization", type=str, default="fp16", choices=["fp16", "int8", "int4"],
                        help="fp16 (default), int8 (Q8_0) or int4 (Q4_0). The Ollama model tag is "
                             "qwen2.5:<size>-{fp16|q8_0|q4_0} and must already be created.")
    args = parser.parse_args()
    run_benchmark(args.model.lower(), args.quantization)
