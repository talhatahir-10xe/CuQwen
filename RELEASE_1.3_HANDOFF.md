# CuQwen Release 1.3 — Qwen3 Benchmarking Handoff

> **You are a fresh Claude Code session running on a RunPod instance with an NVIDIA RTX 3090 (24 GB).**
> This document is your single source of truth. Read it fully before doing anything. It explains
> (1) what CuQwen is and how this repo is laid out, (2) what was already built and validated in
> release 1.3, and (3) the exact job you need to carry out now: **re-verify correctness of every
> CuQwen configuration, benchmark CuQwen / vLLM / llama.cpp on the Qwen3 models, optimize CuQwen
> until it beats both engines, and write `docs/RELEASE_1.3_BENCHMARK.md` with graphs.**

---

## 0. Your environment (RunPod RTX 3090) — read first

- **GPU:** RTX 3090, 24 GB VRAM, sm_86 → always build CuQwen with `-Dgpu_arch=86`.
- **vLLM and llama.cpp are ALREADY INSTALLED.** Do **not** reinstall them. `llama-bench` should already be built under `benchmark/llama.cpp/build/bin/llama-bench` (or `LLAMA_BENCH` is set). Just run the benchmark scripts. If a framework genuinely isn't importable, stop and tell the user rather than reinstalling.
- **Disk is small.** HuggingFace model downloads are large (Qwen3-14B bf16 ≈ 28 GB). You will likely **run out of disk** if you download everything at once. Work **one model size at a time** and **delete what you no longer need**:
  - Delete an HF snapshot: `rm -rf ~/.cache/huggingface/hub/models--Qwen--<RepoName>` (e.g. `models--Qwen--Qwen3-4B`).
  - Delete an exported CuQwen binary after you've benchmarked it: `rm -f weights/model_qwen3_*_<size>.bin`.
  - Recommended per-size flow (minimizes peak disk): download the Qwen3-`<size>` bf16 checkpoint once → export fp16/int8/int4 (all three reuse that one download) → run CuQwen correctness + benchmark for all three → delete the bf16 snapshot and the three `.bin` files → then handle that size's vLLM (AWQ) and llama.cpp (GGUF) downloads → benchmark → delete those too → move to the next size.
- If you hit a **GPU OOM** (not disk), that configuration simply doesn't fit 24 GB at a 32K FP16 KV cache — mark it `✗` (see §7) and move on, exactly as the Qwen2.5 14B FP16/INT8 cells were marked in release 1.2.

---

## 1. What CuQwen is

CuQwen is a from-scratch **C++/CUDA inference engine** for single-user (**batch size 1**) autoregressive **decode** of Qwen models, targeting the performance ceiling with hand-written CUDA kernels. It supports the **Qwen2.5** and (new in 1.3) **Qwen3** dense families in `fp16` (W16A16) and weights-only `int8` (W8A16) / `int4` (W4A16), with an **FP16 KV cache**. The whole per-token forward pass is captured once into a **CUDA graph** and replayed each step.

It consistently beats vLLM and llama.cpp at batch-1 decode on Qwen2.5 (see release 1.2). **Your mission is to make the same true for Qwen3.**

---

## 2. Repository structure

```
CuQwen/
├── include/
│   ├── config.h          # Compile-time model table: families, sizes, dims, tuning knobs. START HERE.
│   ├── kernels.cuh        # Kernel launcher declarations
│   ├── load_weights.h     # LayerWeights / QwenWeights structs
│   ├── model.cuh          # QwenState + qwen_forward declaration
│   └── tokenizer.h
├── src/
│   ├── kernels.cu         # ALL custom CUDA kernels (the heart of the project)
│   ├── model.cu           # Per-layer forward wiring + CUDA-graph capture
│   ├── load_weights.cpp   # Binary weight loader (validates header vs compiled config)
│   ├── main.cu            # Interactive chat REPL
│   └── tokenizer.cpp      # Byte-level BPE (PCRE2) + chat template
├── export_weights.py      # HF safetensors -> CuQwen .bin + tokenizer JSON
├── CMakeLists.txt         # Top-level build (chat engine)
├── benchmark/
│   ├── cuqwen_benchmark.cu # CuQwen decode-throughput benchmark (9 depths, 32K)
│   ├── CMakeLists.txt
│   ├── bench_vllm.py       # vLLM decode benchmark   (UPDATED for Qwen3; UNTESTED — see §9)
│   ├── bench_llamacpp.py   # llama.cpp decode benchmark (UPDATED for Qwen3; UNTESTED — see §9)
│   └── README.md           # How to build/run all three benchmarks
├── docs/
│   ├── RELEASE_1.2_BENCHMARK.md   # Template/structure to mirror for 1.3
│   ├── RELEASE_1.2_OPTIMIZATION.md
│   └── ...                        # 1.0 / 1.1 docs
├── assets/
│   ├── Release1.2_Benchmark/*.png # The 4 graph types you must reproduce for 1.3
│   └── ...
└── cuda_optimizations/   # Frozen v1..v7 optimization snapshots — DO NOT TOUCH
```

**Build/run flag scheme (release 1.3):** two selectors — `-Dmodel=<family>` and `-Dmodelsize=<size>`:

```bash
# Export (from repo root): downloads HF weights, writes weights/model_<arch>_<quant>_<size>.bin + tokenizer
python3 export_weights.py --model=<qwen2.5|qwen3> --modelsize=<size> --quantization=<fp16|int8|int4>

# Build the chat engine
mkdir -p build && cd build
cmake -Dmodel=<qwen2.5|qwen3> -Dmodelsize=<size> -Dquant=<fp16|int8|int4> -Dgpu_arch=86 ..
make -j$(nproc) && cd ..
./build/cuqwen            # interactive chat

# Build + run the CuQwen benchmark
cd benchmark && mkdir -p build && cd build
cmake -Dmodel=<qwen2.5|qwen3> -Dmodelsize=<size> -Dquant=<fp16|int8|int4> -Dgpu_arch=86 ..
make -j$(nproc) && cd ..
./build/cuqwen_benchmark
```

Sizes: **qwen2.5** = `0.5b 1.5b 3b 7b 14b`; **qwen3** = `0.6b 1.7b 4b 8b 14b`. The engine validates family/dims/quant against the binary header, so a mismatched export/build is caught at load.

---

## 3. What release 1.3 added (Qwen3 support) — architecture notes

Qwen3 dense keeps GQA + SwiGLU + pre-norm RMSNorm + RoPE (θ=1e6), but differs from Qwen2.5 in exactly three ways that touch the engine:

1. **Per-head QK-RMSNorm** — a new kernel `qk_norm` (in `src/kernels.cu`) applies RMSNorm over `head_dim` to Q and K (learned `q_norm`/`k_norm` weights, eps 1e-6) **after the QKV projection and before RoPE**. It's launched from `launch_fused_attn_block` only when the norm weights are non-null.
2. **No QKV bias** — `fused_attn_block_kernel` adds the q/k/v bias only when the bias pointer is non-null (Qwen3 passes null).
3. **Decoupled `head_dim` = 128** — the attention width `q_dim = n_heads*head_dim` is independent of `dim` (e.g. Qwen3-0.6B: dim=1024, q_dim=2048). `state.xb` is sized to `max(dim, q_dim)` in `model.cu` for this reason.

Weight I/O: the binary header carries an `arch_id` (field 11; 0=Qwen2.5, 1=Qwen3), validated on load. `export_weights.py` auto-detects QK-norm / bias presence from the checkpoint; tokenizer JSON is per-family (`weights/qwen25_bpe_ranks.json` / `weights/qwen3_bpe_ranks.json`). Tie: Qwen3 `0.6b/1.7b/4b` tie embeddings; `8b/14b` use a separate LM head (stored INT8 for quantized builds, same path as Qwen2.5-7B).

Qwen3 per-size config (already in `include/config.h`): vocab 151936, head_dim 128, n_kv_heads 8, rope_theta 1e6 for all.

| size | dim | inter | layers | q_heads | kv_heads | tie |
|---|---|---|---|---|---|---|
| 0.6B | 1024 | 3072 | 28 | 16 | 8 | ✓ |
| 1.7B | 2048 | 6144 | 28 | 16 | 8 | ✓ |
| 4B | 2560 | 9728 | 36 | 32 | 8 | ✓ |
| 8B | 4096 | 12288 | 36 | 32 | 8 | ✗ |
| 14B | 5120 | 17408 | 40 | 40 | 8 | ✗ |

---

## 4. ⚠️ CRITICAL pre-step before any benchmarking: set the Qwen3 context window to 32K

For development on an 8 GB RTX 2070, **`include/config.h` currently caps `max_seq_len` at 8192 for every Qwen3 size.** The benchmark samples depths up to 32K, so with the 8K cap it will only measure ~4 of the 9 depths. On the 24 GB RTX 3090 you must raise the Qwen3 `max_seq_len` to **32768** before benchmarking (Qwen3's native context is 40960, so 32K needs no RoPE scaling).

Edit `include/config.h` — inside the `#if defined(ARCH_QWEN3)` block, change every `max_seq_len = 8192;` to `max_seq_len = 32768;` (there are 5, one per size). A quick way:

```bash
# Only touches the Qwen3 block values (8192 -> 32768). Verify the diff afterwards.
sed -i 's/max_seq_len       = 8192;/max_seq_len       = 32768;/g' include/config.h
grep -n "max_seq_len" include/config.h   # confirm Qwen3 now 32768, Qwen2.5 still 32768
```

Then **re-export is NOT required** (max_seq_len isn't validated against the header), but you **must rebuild** both the chat engine and the benchmark after changing `config.h`. Leave Qwen2.5 as-is (already 32768).

---

## 5. The job — overview

1. **Correctness** — verify every CuQwen configuration (Qwen2.5 *and* Qwen3) still produces sane chat output. (§6)
2. **Benchmark CuQwen** — gather Qwen3 decode-throughput numbers (all sizes × fp16/int8/int4), 9 depths, 32K. Qwen2.5 CuQwen numbers are already provided (§8) — you do **not** need to re-run Qwen2.5 CuQwen benchmarks. (§7)
3. **Benchmark vLLM + llama.cpp** — on the **Qwen3** models only (Qwen2.5 numbers already provided in §8). (§7)
4. **Compare & optimize** — if CuQwen does **not** beat both vLLM and llama.cpp on a Qwen3 config (as it does on Qwen2.5), optimize the CuQwen kernels, re-verify correctness, and re-benchmark, iterating until CuQwen wins (or you've clearly characterized why not). (§7)
5. **Write `docs/RELEASE_1.3_BENCHMARK.md`** with the 4 standard graphs, covering **both** Qwen2.5 and Qwen3 across all configs. (§8)

Benchmark methodology is fixed and identical across all three engines: **batch 1, decode-only (prefill excluded), FP16 KV cache, 32K context window, throughput sampled at 9 context depths.**

---

## 6. Step 1 — Correctness verification

For each configuration, export → build → run a couple of chat turns → confirm the answer is coherent/correct, then type `exit`. Non-interactively you can pipe prompts, e.g.:

```bash
printf 'What is the capital of France?\nWhat is 15 + 27?\nexit\n' | ./build/cuqwen
```

Expect a correct answer (Paris; 42). **Note: Qwen3 base models emit `<think>…</think>` reasoning before the answer — that is expected**, not a bug. Qwen2.5 Instruct models answer directly.

**Configurations to verify:**
- **Qwen2.5:** `0.5b 1.5b 3b 7b` × `fp16 int8 int4`, plus `14b int4`. (`14b` fp16/int8 don't fit 24 GB → skip.)
- **Qwen3:** `0.6b 1.7b 4b 8b 14b` × `fp16 int8 int4`, skipping only the ones that OOM at 32K (see §7 feasibility).

Already validated on the RTX 2070 before handoff (should just reproduce): Qwen3 `0.6b`/`1.7b` (all three precisions) and `4b` (int8/int4); Qwen2.5 `0.5b int8`, `1.5b int4`, `1.5b fp16`. Everything else is new-to-test on the 3090 (notably Qwen3 `4b fp16`, `8b`, `14b`, which were compile-verified only).

---

## 7. Step 2–4 — Benchmark, compare, optimize

### 7a. Run the benchmarks

**CuQwen** (per Qwen3 config): build `benchmark/` with the matching flags and run `./build/cuqwen_benchmark`. It prints a table of 9 rows — decode speed (tok/s) at context ends **1K, 5K, 9K, 13K, 17K, 21K, 25K, 29K, 32K** (each a timed 0.5K slice). Record the 9 values in that order.

**vLLM** (Qwen3 only): `python3 bench_vllm.py --model qwen3 --modelsize <size> --quant <q>`. Prints median decode tok/s at 9 depths (`500,3500,…,31500`, which align with CuQwen's 1K…32K ends).

**llama.cpp** (Qwen3 only): `python3 bench_llamacpp.py --model qwen3 --modelsize <size> --quant <q>`.

Repo auto-selection and the Qwen3 caveats (official AWQ only ≥4B, no official weights-only INT8, GGUF has Q8_0/Q4_K_M but no q4_0/fp16, `--repo`/`--gguf` overrides) are documented in `benchmark/README.md` §2 — **read it before running these two.**

### 7b. 24 GB feasibility (32K FP16 KV cache)

KV cache at 32K is heavy for Qwen3 (8 KV heads × 128). Rough totals (weights + 32K FP16 KV); treat as guidance and mark `✗` on actual OOM:

| size | fp16 | int8 | int4 |
|---|---|---|---|
| 0.6B | ✓ | ✓ | ✓ |
| 1.7B | ✓ | ✓ | ✓ |
| 4B | ✓ | ✓ | ✓ |
| 8B | ⚠ tight (may ✗) | ✓ | ✓ |
| 14B | ✗ (~28 GB) | ⚠ tight | ✓ |

Use the same `✗` convention as release 1.2 for cells that don't fit. For a 3-engine comparison a cell is only meaningful if **all three** engines can run it; if CuQwen fits but a reference engine can't (or vice-versa), note it.

### 7c. Compare and optimize

For each Qwen3 config, CuQwen should post the **highest** average decode throughput and ideally win at every depth (that's the Qwen2.5 result). Where it doesn't:

- Profile with Nsight Systems/Compute (`nsys`, `ncu`) and the `benchmark/kernel_profile.cu` harness.
- The most likely Qwen3-specific cost is the new `qk_norm` kernel (an extra launch per layer) and the larger KV cache (8 KV heads). Candidate optimizations: fuse `qk_norm` into the RoPE kernel (one block/head already has the head vector), or fuse it into the projection epilogue; revisit the attention tuning knobs in `config.h` (`ATTN_PARTITIONS`, `KSPLIT_WARPS`) for the 8-KV-head geometry on sm_86.
- **After any kernel change: re-run §6 correctness first, then re-benchmark.** Do not report numbers from code that hasn't passed the chat sanity check.
- Read `docs/RELEASE_1.2_OPTIMIZATION.md` and `docs/RELEASE_1.0_OPTIMIZATION_JOURNEY.md` for the optimization playbook and how prior wins were measured.

Iterate until CuQwen beats both engines on the Qwen3 configs (matching the Qwen2.5 story), or until you can clearly explain and document any remaining dead-heat (as was done for Qwen2.5-14B INT4 vs llama.cpp in release 1.2).

---

## 8. Step 5 — Write `docs/RELEASE_1.3_BENCHMARK.md` with graphs

Mirror `docs/RELEASE_1.2_BENCHMARK.md` (read it for tone/structure). The 1.3 doc must cover **both Qwen2.5 and Qwen3**, all configs, and include the **same four graph types** that `RELEASE_1.2_BENCHMARK.md` references:

1. **Throughput vs. context length (context decay)** — per-config curves, CuQwen vs vLLM vs llama.cpp across the 9 depths.
2. **Average throughput across the 32K window** — bar chart, averaged over the 9 depths.
3. **Average speedup advantage** — CuQwen's % lead over each engine.
4. **Performance decay rate** — drop from first (1K) to last (32K) depth: `(speed_1K − speed_32K)/speed_1K × 100`.

No plotting script is committed — **write a matplotlib script** (put it in `benchmark/` or a scratch dir) that reads the collected numbers and saves PNGs to **`assets/Release1.3_Benchmark/`**, then reference them from the doc with relative paths (`../assets/Release1.3_Benchmark/<name>.png`), the same way `RELEASE_1.2_BENCHMARK.md` does. Match the visual style of the existing `assets/Release1.2_Benchmark/*.png`. Also add a Qwen3 results section to the main `README.md` performance table once numbers exist.

### 8a. Qwen2.5 reference numbers (already measured — use directly, do NOT re-run Qwen2.5)

Depth order is the 9 sampled context depths **[1K, 5K, 9K, 13K, 17K, 21K, 25K, 29K, 32K]**. Units = decode tok/s. Each row is CuQwen / vLLM / llama.cpp.

**Qwen2.5-0.5B**
- int4 — CuQwen: `803, 770, 718, 690, 649, 625, 597, 571, 556` · vLLM: `550, 541, 532, 482, 474, 466, 435, 415, 408` · llama.cpp: `647, 624, 572, 545, 515, 509, 488, 473, 451`
- int8 — CuQwen: `720, 690, 646, 621, 590, 571, 547, 528, 514` · vLLM: `493, 486, 481, 443, 428, 410, 390, 383, 374` · llama.cpp: `589, 562, 518, 499, 482, 459, 446, 429, 422`
- fp16 — CuQwen: `586, 567, 537, 521, 498, 484, 466, 452, 442` · vLLM: `454, 447, 442, 408, 399, 392, 377, 360, 350` · llama.cpp: `442, 433, 414, 397, 389, 376, 364, 351, 343`

**Qwen2.5-1.5B**
- int4 — CuQwen: `454, 429, 399, 381, 355, 338, 319, 307, 297` · vLLM: `354, 340, 321, 299, 289, 263, 248, 234, 228` · llama.cpp: `413, 390, 373, 353, 336, 320, 305, 292, 280`
- int8 — CuQwen: `353, 338, 319, 308, 290, 280, 266, 258, 252` · vLLM: `282, 278, 259, 243, 235, 220, 210, 199, 195` · llama.cpp: `308, 297, 286, 271, 262, 252, 244, 237, 229`
- fp16 — CuQwen: `236, 230, 221, 216, 207, 202, 194, 190, 187` · vLLM: `196, 191, 185, 174, 171, 164, 158, 152, 150` · llama.cpp: `207, 202, 196, 191, 185, 181, 177, 172, 166`

**Qwen2.5-3B**
- int4 — CuQwen: `283, 270, 254, 245, 231, 223, 212, 205, 199` · vLLM: `234, 225, 213, 201, 194, 179, 170, 162, 158` · llama.cpp: `264, 255, 242, 231, 222, 214, 206, 199, 191`
- int8 — CuQwen: `201, 195, 187, 182, 174, 169, 163, 159, 155` · vLLM: `171, 166, 160, 152, 149, 140, 134, 129, 127` · llama.cpp: `186, 180, 174, 168, 163, 159, 154, 151, 146`
- fp16 — CuQwen: `126, 123, 120, 118, 115, 113, 109, 108, 106` · vLLM: `110, 107, 105, 103, 101, 97, 94, 91, 90` · llama.cpp: `117, 115, 112, 110, 107, 106, 104, 101, 100`

**Qwen2.5-7B**
- int4 — CuQwen: `173, 165, 155, 149, 141, 135, 129, 125, 122` · vLLM: `149, 143, 136, 128, 123, 117, 113, 109, 105` · llama.cpp: `160, 154, 147, 142, 136, 132, 127, 123, 119`
- int8 — CuQwen: `107, 104, 100, 98, 94, 92, 89, 87, 85` · vLLM: `96, 94, 91, 87, 84, 82, 80, 78, 76` · llama.cpp: `100, 98, 95, 92, 90, 88, 86, 84, 82`
- fp16 — CuQwen: `59, 58, 57, 56, 55, 54, 53, 52, 51` · vLLM: `52, 51, 50, 49, 48, 47, 46, 45, 44` · llama.cpp: `57, 56, 55, 54, 53, 52, 51, 50, 49`

**Qwen2.5-14B** (int4 only — fp16/int8 don't fit 24 GB)
- int4 — CuQwen: `87, 81, 74, 70, 64, 60, 57, 54, 52` · vLLM: `82, 75, 69, 64, 61, 57, 53, 51, 48` · llama.cpp: `82, 78, 73, 68, 65, 61, 57, 55, 52`

### 8b. Qwen3 numbers — you will collect these

Record them in the identical layout (per size × precision, CuQwen/vLLM/llama.cpp, 9 depths) as you run §7. Keep a machine-readable copy (e.g. a JSON or CSV in `benchmark/`) so the plotting script can consume both the Qwen2.5 table above and your new Qwen3 data.

---

## 9. Known issues & caveats

- **The vLLM and llama.cpp benchmark scripts were updated for Qwen3 but NOT executed** (vLLM/llama.cpp weren't installed on the dev machine). They byte-compile and mirror the working Qwen2.5 logic, but a runtime bug is possible — **it's rare, but if a script errors, fix it** (the Qwen3 repo/quant selection and arg handling are the most likely spots). The Qwen2.5 path in both scripts is unchanged from the working 1.2 version.
- **Qwen3 vLLM `int8`:** no official weights-only (W8A16) Qwen3 repo exists (Qwen ships FP8 = W8A8). `bench_vllm.py` will exit asking for `--repo <gptq-int8-build>`. For a fair weights-only INT8 comparison you'll need a third-party GPTQ-Int8 checkpoint, or document INT8 vLLM as N/A for Qwen3.
- **Qwen3 vLLM `int4`:** official AWQ exists only for `4b/8b/14b` (`Qwen/Qwen3-<S>-AWQ`); for `0.6b/1.7b` pass `--repo`.
- **Qwen3 llama.cpp:** official GGUF has `Q8_0` and `Q4_K_M` (not legacy `q4_0`) and **no fp16** file; `0.6b/1.7b` ship only `Q8_0`. Use `--gguf PATH` to benchmark any other GGUF you convert yourself.
- `cuda_optimizations/v1..v7/` are frozen historical snapshots — never build or modify them.

---

## 10. Current state at handoff

- **Branch:** `release1.3` (all 1.3 work lives here). The code is **implemented but not committed** — the user commits. Do your work on this branch; commit only if the user asks.
- **Validated on RTX 2070 (8 GB):** Qwen3 `0.6b` (fp16/int8/int4), `1.7b` (fp16/int8/int4), `4b` (int8/int4); Qwen3 `8b`/`14b` compile-verified only. Qwen2.5 regression: `0.5b int8`, `1.5b int4`, `1.5b fp16` — all correct.
- **Not yet done (your job):** Qwen3 benchmarking at 32K on the 3090, the vLLM/llama.cpp Qwen3 runs, any needed optimization, and `docs/RELEASE_1.3_BENCHMARK.md` + graphs.

### Suggested order of operations
1. `git checkout release1.3`, skim `include/config.h`, `src/kernels.cu` (`qk_norm_kernel`, `fused_attn_block`), `src/model.cu`.
2. Do §4 (set Qwen3 `max_seq_len = 32768`, rebuild).
3. §6 correctness pass (spot-check a few Qwen2.5 + every Qwen3 config that fits).
4. §7 benchmarks — one model size at a time, deleting downloads/binaries as you go (§0).
5. §7c optimize if CuQwen isn't winning; re-verify + re-benchmark.
6. §8 write the doc + graphs.
7. Report results to the user; ask before committing.
