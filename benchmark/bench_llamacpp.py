#!/usr/bin/env python3
"""
Qwen2.5 / Qwen3 decode-speed benchmark on llama.cpp (weight-only GGUF, FP16 KV cache).

Usage:
    python3 bench_llamacpp.py --model qwen2.5 --modelsize 3b --quant int4
    python3 bench_llamacpp.py --model qwen3   --modelsize 8b --quant int8
    python3 bench_llamacpp.py --model qwen3   --modelsize 4b --quant int4 --gguf /path/to.gguf

--model     : qwen2.5 | qwen3            (model family / architecture)
--modelsize : qwen2.5 -> 0.5b|1.5b|3b|7b|14b ; qwen3 -> 0.6b|1.7b|4b|8b|14b
--quant     : fp16 | int8 | int4         ->  GGUF full / q8_0 / 4-bit
--gguf      : optional explicit path to a .gguf (skips the auto-download)

Wraps the native `llama-bench` tool (built with CUDA + CUDA graphs + FlashAttention).
For each of 9 context depths it prefills that many tokens, decodes DECODE_TOKENS
tokens, and records the decode-only speed (llama.cpp's tg rate). All layers on GPU
(-ngl 99), FlashAttention on (-fa 1), KV cache FP16 (llama.cpp default). GGUF weights
are pulled from the official Qwen GGUF repos on first use. Results print to stdout.

GGUF quant suffix by family (official Qwen GGUF repos):
  Qwen2.5  (Qwen/Qwen2.5-<S>-Instruct-GGUF):  fp16 -> fp16 | int8 -> q8_0 | int4 -> q4_0
  Qwen3    (Qwen/Qwen3-<S>-GGUF):              int8 -> q8_0 | int4 -> q4_k_m
    * Qwen3 official GGUF ships no full-precision (fp16) file and no legacy q4_0;
      the closest official 4-bit is Q4_K_M. 0.6B/1.7B ship only Q8_0.
    * Pass --gguf PATH to benchmark any other GGUF (e.g. a fp16/q4_0 build you made).
GGUF filenames are matched case-insensitively (Qwen3 uses upper-case Q8_0 / Q4_K_M).
"""
import argparse, glob, json, os, re, statistics, subprocess, sys

WINDOWS = [(500, 1000), (3500, 4000), (7500, 8000), (11500, 12000), (15500, 16000),
           (19500, 20000), (23500, 24000), (27500, 28000), (31500, 32000)]

SIZES = {
    "qwen2.5": {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B", "14b": "14B"},
    "qwen3":   {"0.6b": "0.6B", "1.7b": "1.7B", "4b": "4B", "8b": "8B", "14b": "14B"},
}

# Family -> { quant -> GGUF filename suffix }. Only combos with an official file.
QSUF = {
    "qwen2.5": {"fp16": "fp16", "int8": "q8_0", "int4": "q4_0"},
    "qwen3":   {"int8": "q8_0", "int4": "q4_k_m"},
}

HERE = os.path.dirname(os.path.abspath(__file__))

DECODE_TOKENS = 500       # tokens decoded per depth window
REPEATS       = 3         # repeats per depth (median reported)
GPU_LAYERS    = 99        # layers offloaded to GPU (99 = all)


def gguf_repo(family, size):
    S = SIZES[family][size]
    return (f"Qwen/Qwen2.5-{S}-Instruct-GGUF" if family == "qwen2.5"
            else f"Qwen/Qwen3-{S}-GGUF")


def find_llama_bench(explicit):
    for c in [explicit, os.environ.get("LLAMA_BENCH"),
              os.path.join(HERE, "llama.cpp/build/bin/llama-bench"),
              os.path.join(HERE, "llama.cpp/build/llama-bench")]:
        if c and os.path.isfile(c):
            return c
    sys.exit("llama-bench not found. Build it (see README) or pass --llama-bench PATH.")


def ensure_gguf(family, size, quant):
    from huggingface_hub import snapshot_download
    if quant not in QSUF[family]:
        avail = ", ".join(QSUF[family])
        sys.exit(f"No official {family} GGUF for '{quant}'. Available: {avail}. "
                 f"Pass --gguf PATH to use a custom GGUF.")
    repo = gguf_repo(family, size)
    suf = QSUF[family][quant]
    # Qwen2.5 GGUF files are lower-case (…-q4_0.gguf); Qwen3 are upper-case
    # (…-Q4_K_M.gguf). Fetch both casings, then match case-insensitively.
    d = snapshot_download(repo, allow_patterns=[f"*{suf}*.gguf", f"*{suf.upper()}*.gguf"])
    files = sorted(f for f in glob.glob(os.path.join(d, "*.gguf"))
                   if suf.lower() in os.path.basename(f).lower())
    if not files:
        sys.exit(f"No GGUF matching *{suf}*.gguf (any case) in {repo}")
    first = [f for f in files if re.search(r"-0*1-of-\d+\.gguf$", f)]
    return first[0] if first else files[0]


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5 / Qwen3 llama.cpp decode benchmark")
    ap.add_argument("--model", required=True, choices=list(SIZES),
                    help="model family (qwen2.5 or qwen3)")
    ap.add_argument("--modelsize", required=True, help="size within the family")
    ap.add_argument("--quant", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--gguf", default=None, help="explicit .gguf path (skips auto-download)")
    ap.add_argument("--llama-bench", default=None, help="path to the llama-bench binary")
    args = ap.parse_args()

    if args.modelsize not in SIZES[args.model]:
        valid = ", ".join(SIZES[args.model])
        ap.error(f"invalid --modelsize '{args.modelsize}' for {args.model}; valid: {valid}")

    bin = find_llama_bench(args.llama_bench)
    starts = [w[0] for w in WINDOWS]

    gguf = args.gguf or ensure_gguf(args.model, args.modelsize, args.quant)
    print(f"[llamacpp] {args.model}/{args.modelsize}/{args.quant} -> {os.path.basename(gguf)}", flush=True)

    depths = ",".join(str(s) for s in starts)
    cmd = [bin, "-m", gguf, "-ngl", str(GPU_LAYERS), "-fa", "1",
           "-p", "0", "-n", str(DECODE_TOKENS), "-d", depths,
           "-r", str(REPEATS), "-o", "json"]
    print("[llamacpp] " + " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr[-2000:])
        sys.exit(f"llama-bench failed ({res.returncode})")

    rows = json.loads(res.stdout)
    by_depth = {}
    for r in rows:
        if int(r.get("n_gen", 0)) <= 0:
            continue
        by_depth[int(r.get("n_depth", 0))] = float(r["avg_ts"])

    print(f"\n[llamacpp] {args.model}/{args.modelsize}/{args.quant} decode throughput (tok/s):")
    depth_tps = []
    for s in starts:
        tps = by_depth.get(s)
        depth_tps.append((s, tps))
        print(f"  depth {s:>6}: {tps:.2f}" if tps else f"  depth {s:>6}: n/a")

    valid = [t for _, t in depth_tps if t]
    if valid:
        avg = statistics.mean(valid)
        decay = (valid[0] - valid[-1]) / valid[0] * 100 if valid[0] else 0.0
        print(f"  average : {avg:.2f} tok/s")
        print(f"  decay   : {decay:.1f}% (first -> last depth)")


if __name__ == "__main__":
    main()
