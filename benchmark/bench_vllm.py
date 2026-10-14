#!/usr/bin/env python3
"""
Qwen2.5 / Qwen3 decode-speed benchmark on vLLM (weight-only quant, FP16 KV cache).

Usage:
    python3 bench_vllm.py --model qwen2.5 --modelsize 3b --quant int4
    python3 bench_vllm.py --model qwen3   --modelsize 8b --quant int4
    python3 bench_vllm.py --model qwen3   --modelsize 4b --quant fp16
    python3 bench_vllm.py --model qwen3   --modelsize 8b --quant int8 --repo <hf_repo>

--model     : qwen2.5 | qwen3            (model family / architecture)
--modelsize : qwen2.5 -> 0.5b|1.5b|3b|7b|14b ; qwen3 -> 0.6b|1.7b|4b|8b|14b
--quant     : fp16 | int8 | int4         (W16A16 / W8A16 / W4A16, weights only)
--repo      : optional HF repo override (use any checkpoint for the chosen combo)

For each of 9 context depths it prefills exactly `start` tokens, then decodes
DECODE_TOKENS tokens and reports the *decode-only* speed (tokens/s) — vLLM's
first->last generated-token interval, which excludes prefill. The KV cache stays
FP16 (kv_cache_dtype follows the float16 model dtype). Results are printed to
stdout.

The repository is auto-selected from --model / --modelsize / --quant:

  Qwen2.5 (official weights-only GPTQ, as in the 1.2 benchmark):
    fp16 -> Qwen/Qwen2.5-<S>-Instruct              (dtype float16, W16A16)
    int8 -> Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int8    (GPTQ 8-bit weights, W8A16)
    int4 -> Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int4    (GPTQ 4-bit weights, W4A16)

  Qwen3 (official Qwen quantizations differ from Qwen2.5):
    fp16 -> Qwen/Qwen3-<S>                          (dtype float16, W16A16)
    int4 -> Qwen/Qwen3-<S>-AWQ                      (AWQ 4-bit weights, W4A16)
            *only published for 4B/8B/14B; for 0.6B/1.7B pass --repo*
    int8 -> no official weights-only (W8A16) repo exists for Qwen3
            (Qwen ships FP8 = W8A8, a different regime) -> pass --repo <gptq-int8 build>

Pass --repo to override the auto-selection with any compatible checkpoint.
"""
import argparse, statistics, time

WINDOWS = [(500, 1000), (3500, 4000), (7500, 8000), (11500, 12000), (15500, 16000),
           (19500, 20000), (23500, 24000), (27500, 28000), (31500, 32000)]

# Family -> { size flag -> HF size tag }.
SIZES = {
    "qwen2.5": {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B", "14b": "14B"},
    "qwen3":   {"0.6b": "0.6B", "1.7b": "1.7B", "4b": "4B", "8b": "8B", "14b": "14B"},
}

DECODE_TOKENS = 500       # tokens decoded per depth window
REPEATS       = 3         # repeats per depth (median reported)
WARMUP        = 3         # warmup generations before timing
MAX_MODEL_LEN = 32768     # context window
GPU_MEM_UTIL  = 0.90      # vLLM GPU memory fraction


def vllm_repo(family, size, quant):
    """Auto-select the HF repo for a (family, size, quant). Raises with guidance
    when no official weights-only checkpoint exists for the combination."""
    S = SIZES[family][size]
    if family == "qwen2.5":
        base = f"Qwen/Qwen2.5-{S}-Instruct"
        return {"fp16": base, "int8": base + "-GPTQ-Int8", "int4": base + "-GPTQ-Int4"}[quant]

    # qwen3
    if quant == "fp16":
        return f"Qwen/Qwen3-{S}"
    if quant == "int4":
        # Official Qwen3 weights-only 4-bit is AWQ (W4A16), published only for >=4B.
        if size in ("0.6b", "1.7b"):
            raise SystemExit(
                f"[vllm] No official Qwen3 AWQ (int4) repo for {size}. "
                f"Pass --repo <hf_repo> with a 4-bit build (e.g. a third-party GPTQ/AWQ).")
        return f"Qwen/Qwen3-{S}-AWQ"
    if quant == "int8":
        raise SystemExit(
            "[vllm] Qwen3 has no official weights-only INT8 (W8A16) checkpoint "
            "(Qwen ships FP8 = W8A8, a different regime). "
            "Pass --repo <hf_repo> with a GPTQ-Int8 build, or benchmark int4/fp16.")
    raise SystemExit(f"[vllm] Unsupported quant '{quant}'.")


def build_ids(tok, n):
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "Benchmarks require many tokens to fill the context window. ",
                      add_special_tokens=False)
    ids = (base * (n // len(base) + 1))[:n]
    assert len(ids) == n
    return ids


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5 / Qwen3 vLLM decode benchmark")
    ap.add_argument("--model", required=True, choices=list(SIZES),
                    help="model family (qwen2.5 or qwen3)")
    ap.add_argument("--modelsize", required=True, help="size within the family")
    ap.add_argument("--quant", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--repo", default=None,
                    help="override the auto-selected HF repo for this combo")
    args = ap.parse_args()

    if args.modelsize not in SIZES[args.model]:
        valid = ", ".join(SIZES[args.model])
        ap.error(f"invalid --modelsize '{args.modelsize}' for {args.model}; valid: {valid}")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    repo = args.repo or vllm_repo(args.model, args.modelsize, args.quant)

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    print(f"[vllm] {args.model}/{args.modelsize}/{args.quant} -> {repo} (dtype=float16, "
          f"max_model_len={MAX_MODEL_LEN})", flush=True)
    t0 = time.time()
    llm = LLM(model=repo, dtype="float16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEM_UTIL, enforce_eager=False,
              enable_prefix_caching=False, disable_log_stats=False,  # stats -> per-request timing
              max_num_seqs=1, seed=0)
    print(f"[vllm] model ready in {time.time()-t0:.1f}s", flush=True)

    def run_one(start, gen):
        ids = build_ids(tok, start)
        sp = SamplingParams(temperature=0.0, max_tokens=gen, min_tokens=gen,
                            ignore_eos=True, detokenize=False)
        o = llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)[0]
        n_gen = len(o.outputs[0].token_ids)
        m = getattr(o, "metrics", None)
        tps = None
        if m is not None and getattr(m, "first_token_ts", None) and getattr(m, "last_token_ts", None):
            dur = m.last_token_ts - m.first_token_ts
            steps = (m.num_generation_tokens or n_gen) - 1
            if dur > 0 and steps > 0:
                tps = steps / dur
        return tps

    def run_subtraction(start, gen):  # fallback if metrics unavailable
        ids = build_ids(tok, start)
        base = dict(temperature=0.0, ignore_eos=True, detokenize=False)
        t = time.perf_counter()
        llm.generate({"prompt_token_ids": ids}, SamplingParams(max_tokens=1, min_tokens=1, **base), use_tqdm=False)
        t1 = time.perf_counter() - t
        t = time.perf_counter()
        llm.generate({"prompt_token_ids": ids}, SamplingParams(max_tokens=gen + 1, min_tokens=gen + 1, **base), use_tqdm=False)
        t2 = time.perf_counter() - t
        d = t2 - t1
        return (gen / d if d > 0 else None)

    print(f"[vllm] warmup x{WARMUP} ...", flush=True)
    for _ in range(WARMUP):
        run_one(512, DECODE_TOKENS)

    depth_medians = []
    for (start, end) in WINDOWS:
        runs = []
        for r in range(REPEATS):
            tps = run_one(start, DECODE_TOKENS)
            if tps is None:
                tps = run_subtraction(start, DECODE_TOKENS)
            runs.append(tps)
            print(f"[vllm] {args.model}/{args.modelsize}/{args.quant} win {start}-{start+DECODE_TOKENS} "
                  f"run {r+1}/{REPEATS}: {tps:.2f} tok/s", flush=True)
        tpss = [x for x in runs if x]
        median = statistics.median(tpss) if tpss else None
        depth_medians.append((start, median))

    print(f"\n[vllm] {args.model}/{args.modelsize}/{args.quant} decode throughput (median tok/s):")
    for start, median in depth_medians:
        print(f"  depth {start:>6}: {median:.2f}" if median else f"  depth {start:>6}: n/a")
    valid = [m for _, m in depth_medians if m]
    if valid:
        avg = statistics.mean(valid)
        decay = (valid[0] - valid[-1]) / valid[0] * 100 if valid[0] else 0.0
        print(f"  average : {avg:.2f} tok/s")
        print(f"  decay   : {decay:.1f}% (first -> last depth)")


if __name__ == "__main__":
    main()
