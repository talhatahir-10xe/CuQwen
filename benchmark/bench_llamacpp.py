#!/usr/bin/env python3
"""
Qwen2.5 decode-speed benchmark on llama.cpp (weight-only GGUF quant, FP16 KV cache).

Usage:
    python3 bench_llamacpp.py --model 3b --quant int4
    python3 bench_llamacpp.py --model 7b --quant fp16 --repeats 3

--model : 0.5b | 1.5b | 3b | 7b
--quant : fp16 | int8 | int4      ->  GGUF fp16 / q8_0 / q4_0 (W16A16 / ~W8A16 / ~W4A16)

Wraps the native `llama-bench` tool (built with CUDA + CUDA graphs + FlashAttention).
For each of 9 context depths it prefills that many tokens, decodes `--decode-tokens`
(default 500), and records the decode-only speed (llama.cpp's tg rate). All layers on
GPU (-ngl 99), FlashAttention on (-fa 1), KV cache FP16 (llama.cpp default). GGUF
weights are pulled from the official Qwen GGUF repos on first use (sharded files are
downloaded whole; llama-bench loads them from the first shard).
"""
import argparse, glob, json, os, re, statistics, subprocess, sys, time

WINDOWS = [(500, 1000), (3500, 4000), (7500, 8000), (11500, 12000), (15500, 16000),
           (19500, 20000), (23500, 24000), (27500, 28000), (31500, 32000)]
SIZES = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B"}
QSUF = {"fp16": "fp16", "int8": "q8_0", "int4": "q4_0"}
HERE = os.path.dirname(os.path.abspath(__file__))


def find_llama_bench(explicit):
    for c in [explicit, os.environ.get("LLAMA_BENCH"),
              os.path.join(HERE, "llama.cpp/build/bin/llama-bench"),
              os.path.join(HERE, "llama.cpp/build/llama-bench")]:
        if c and os.path.isfile(c):
            return c
    sys.exit("llama-bench not found. Build it (see setup.md) or pass --llama-bench PATH.")


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
    ap.add_argument("--out", default=None)
    ap.add_argument("--decode-tokens", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu-layers", type=int, default=99)
    ap.add_argument("--llama-bench", default=None)
    ap.add_argument("--windows", default="")
    args = ap.parse_args()

    bin = find_llama_bench(args.llama_bench)
    out = args.out or f"results/llamacpp_{args.model}_{args.quant}.json"
    starts = [w[0] for w in WINDOWS]
    if args.windows:
        keep = {int(x) for x in args.windows.split(",")}
        starts = [s for s in starts if s in keep]

    gguf = ensure_gguf(args.model, args.quant)
    print(f"[llamacpp] {args.model}/{args.quant} -> {os.path.basename(gguf)}", flush=True)

    depths = ",".join(str(s) for s in starts)
    cmd = [bin, "-m", gguf, "-ngl", str(args.gpu_layers), "-fa", "1",
           "-p", "0", "-n", str(args.decode_tokens), "-d", depths,
           "-r", str(args.repeats), "-o", "json"]
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
        by_depth[int(r.get("n_depth", 0))] = (float(r["avg_ts"]), float(r.get("stddev_ts", 0.0)))

    results = []
    for s in starts:
        mean, std = by_depth.get(s, (None, None))
        if mean is not None:
            print(f"[llamacpp] {args.model}/{args.quant} depth {s}: {mean:.2f} tok/s", flush=True)
        results.append(dict(target_start=s, target_end=s + args.decode_tokens,
                            decode_tokens=args.decode_tokens, actual_prefill_tokens=s,
                            decode_tps_mean=mean, decode_tps_median=mean, decode_tps_std=std,
                            runs=[dict(decode_tps=mean, source="llama-bench")]))

    payload = dict(framework="llamacpp", model_size=args.model, level=args.quant,
                   model=os.path.basename(gguf), gpu="cuda", kv_cache_dtype="fp16",
                   flash_attn=True, cuda_graphs=True, decode_tokens=args.decode_tokens,
                   repeats=args.repeats, tool="llama-bench", timestamp=time.time(), windows=results)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(f"[llamacpp] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
