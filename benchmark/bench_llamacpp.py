#!/usr/bin/env python3
"""
Qwen2.5 decode-speed benchmark on llama.cpp (weight-only GGUF quant, FP16 KV cache).

Usage:
    python3 bench_llamacpp.py --model 3b --quant int4
    python3 bench_llamacpp.py --model 7b --quant fp16

--model : 0.5b | 1.5b | 3b | 7b
--quant : fp16 | int8 | int4      ->  GGUF fp16 / q8_0 / q4_0 (W16A16 / ~W8A16 / ~W4A16)

Wraps the native `llama-bench` tool (built with CUDA + CUDA graphs + FlashAttention).
For each of 9 context depths it prefills that many tokens, decodes DECODE_TOKENS
tokens, and records the decode-only speed (llama.cpp's tg rate). All layers on GPU
(-ngl 99), FlashAttention on (-fa 1), KV cache FP16 (llama.cpp default). GGUF weights
are pulled from the official Qwen GGUF repos on first use (sharded files are
downloaded whole; llama-bench loads them from the first shard). Results are printed
to stdout.
"""
import argparse, glob, json, os, re, statistics, subprocess, sys

WINDOWS = [(500, 1000), (3500, 4000), (7500, 8000), (11500, 12000), (15500, 16000),
           (19500, 20000), (23500, 24000), (27500, 28000), (31500, 32000)]
SIZES = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B"}
QSUF = {"fp16": "fp16", "int8": "q8_0", "int4": "q4_0"}
HERE = os.path.dirname(os.path.abspath(__file__))

DECODE_TOKENS = 500       # tokens decoded per depth window
REPEATS       = 3         # repeats per depth (median reported)
GPU_LAYERS    = 99        # layers offloaded to GPU (99 = all)


def find_llama_bench(explicit):
    for c in [explicit, os.environ.get("LLAMA_BENCH"),
              os.path.join(HERE, "llama.cpp/build/bin/llama-bench"),
              os.path.join(HERE, "llama.cpp/build/llama-bench")]:
        if c and os.path.isfile(c):
            return c
    sys.exit("llama-bench not found. Build it (see README) or pass --llama-bench PATH.")


def ensure_gguf(size, quant):
    from huggingface_hub import snapshot_download
    repo = f"Qwen/Qwen2.5-{SIZES[size]}-Instruct-GGUF"
    suf = QSUF[quant]
    d = snapshot_download(repo, allow_patterns=[f"*{suf}*.gguf"])
    files = sorted(glob.glob(os.path.join(d, f"*{suf}*.gguf")))
    if not files:
        sys.exit(f"No GGUF matching *{suf}*.gguf in {repo}")
    first = [f for f in files if re.search(r"-0*1-of-\d+\.gguf$", f)]
    return first[0] if first else files[0]


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5 llama.cpp decode benchmark")
    ap.add_argument("--model", required=True, choices=list(SIZES))
    ap.add_argument("--quant", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--llama-bench", default=None, help="path to the llama-bench binary")
    args = ap.parse_args()

    bin = find_llama_bench(args.llama_bench)
    starts = [w[0] for w in WINDOWS]

    gguf = ensure_gguf(args.model, args.quant)
    print(f"[llamacpp] {args.model}/{args.quant} -> {os.path.basename(gguf)}", flush=True)

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

    print(f"\n[llamacpp] {args.model}/{args.quant} decode throughput (tok/s):")
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
