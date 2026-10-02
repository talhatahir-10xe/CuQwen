#!/usr/bin/env python3
"""
Qwen2.5 decode-speed benchmark on vLLM (weight-only quant, FP16 KV cache).

Usage:
    python3 bench_vllm.py --model 3b --quant int4
    python3 bench_vllm.py --model 7b --quant fp16 --repeats 3

--model : 0.5b | 1.5b | 3b | 7b     (Qwen2.5-*-Instruct)
--quant : fp16 | int8 | int4        (W16A16 / W8A16 / W4A16, weights only)

For each of 9 context "patches" it prefills exactly `start` tokens, then decodes
`--decode-tokens` (default 500) tokens and reports the *decode-only* speed
(tokens/s) — vLLM's first->last generated-token interval, which excludes prefill.
KV cache stays FP16 (kv_cache_dtype follows the float16 model dtype).

Quantization is auto-detected from each repo's config:
    fp16 -> Qwen/Qwen2.5-<S>-Instruct              (dtype float16)
    int8 -> Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int8    (GPTQ 8-bit weights, W8A16)
    int4 -> Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int4    (GPTQ 4-bit weights, W4A16)
"""
import argparse, json, os, statistics, time

WINDOWS = [(500, 1000), (3500, 4000), (7500, 8000), (11500, 12000), (15500, 16000),
           (19500, 20000), (23500, 24000), (27500, 28000), (31500, 32000)]
SIZES = {"0.5b": "0.5B", "1.5b": "1.5B", "3b": "3B", "7b": "7B"}


def vllm_repo(size, quant):
    base = f"Qwen/Qwen2.5-{SIZES[size]}-Instruct"
    return {"fp16": base, "int8": base + "-GPTQ-Int8", "int4": base + "-GPTQ-Int4"}[quant]


def build_ids(tok, n):
    base = tok.encode("The quick brown fox jumps over the lazy dog. "
                      "Benchmarks require many tokens to fill the context window. ",
                      add_special_tokens=False)
    ids = (base * (n // len(base) + 1))[:n]
    assert len(ids) == n
    return ids


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5 vLLM decode benchmark")
    ap.add_argument("--model", required=True, choices=list(SIZES))
    ap.add_argument("--quant", required=True, choices=["fp16", "int8", "int4"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--decode-tokens", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--windows", default="", help="comma list of start indices (smoke test)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    repo = vllm_repo(args.model, args.quant)
    out = args.out or f"results/vllm_{args.model}_{args.quant}.json"
    windows = WINDOWS
    if args.windows:
        keep = {int(x) for x in args.windows.split(",")}
        windows = [w for w in WINDOWS if w[0] in keep]

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    print(f"[vllm] {args.model}/{args.quant} -> {repo} (dtype=float16, "
          f"max_model_len={args.max_model_len})", flush=True)
    t0 = time.time()
    llm = LLM(model=repo, dtype="float16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=False,
              enable_prefix_caching=False, disable_log_stats=False,  # stats -> per-request timing
              max_num_seqs=1, seed=0)
    print(f"[vllm] model ready in {time.time()-t0:.1f}s", flush=True)

    def run_one(start, gen):
        ids = build_ids(tok, start)
        sp = SamplingParams(temperature=0.0, max_tokens=gen, min_tokens=gen,
                            ignore_eos=True, detokenize=False)
        w0 = time.perf_counter()
        o = llm.generate({"prompt_token_ids": ids}, sp, use_tqdm=False)[0]
        wall = time.perf_counter() - w0
        n_gen = len(o.outputs[0].token_ids)
        m = getattr(o, "metrics", None)
        tps = dt = src = None
        if m is not None and getattr(m, "first_token_ts", None) and getattr(m, "last_token_ts", None):
            dur = m.last_token_ts - m.first_token_ts
            steps = (m.num_generation_tokens or n_gen) - 1
            if dur > 0 and steps > 0:
                dt, tps, src = dur, steps / dur, "metrics"
        return dict(prompt_tokens=len(ids), gen_tokens=n_gen, wall_s=wall,
                    decode_time_s=dt, decode_tps=tps, source=src)

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
        return (gen / d if d > 0 else None), d

    print(f"[vllm] warmup x{args.warmup} ...", flush=True)
    for _ in range(args.warmup):
        run_one(512, min(args.decode_tokens, 500))

    results = []
    for (start, end) in windows:
        gen = args.decode_tokens
        runs = []
        for r in range(args.repeats):
            info = run_one(start, gen)
            if info["decode_tps"] is None:
                tps, dt = run_subtraction(start, gen)
                info.update(decode_tps=tps, decode_time_s=dt, source="subtraction")
            runs.append(info)
            print(f"[vllm] {args.model}/{args.quant} win {start}-{start+gen} "
                  f"run {r+1}/{args.repeats}: {info['decode_tps']:.2f} tok/s ({info['source']})", flush=True)
        tpss = [x["decode_tps"] for x in runs if x["decode_tps"]]
        results.append(dict(target_start=start, target_end=start + gen, decode_tokens=gen,
                            actual_prefill_tokens=runs[0]["prompt_tokens"],
                            decode_tps_mean=statistics.mean(tpss) if tpss else None,
                            decode_tps_median=statistics.median(tpss) if tpss else None,
                            decode_tps_std=statistics.pstdev(tpss) if len(tpss) > 1 else 0.0,
                            runs=runs))

    payload = dict(framework="vllm", model_size=args.model, level=args.quant, model=repo,
                   gpu="cuda", kv_cache_dtype="fp16", activation_dtype="fp16",
                   max_model_len=args.max_model_len, decode_tokens=args.decode_tokens,
                   repeats=args.repeats, timestamp=time.time(), windows=results)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(payload, open(out, "w"), indent=2)
    print(f"[vllm] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
