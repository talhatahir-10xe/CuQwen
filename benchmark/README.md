# CuQwen Benchmarking Suite

This directory contains the benchmark applications and scripts used to compare **CuQwen** against production LLM inference frameworks (**vLLM** and **llama.cpp**).

All benchmarks measure bare-metal, single-user (`Batch Size 1`) autoregressive **decode throughput** (tokens/s) as the KV cache grows across the **32K context window**. Only the weights are quantized; the **KV cache stays FP16** on every engine. CuQwen supports `fp16` (W16A16), `int8` (W8A16) and `int4` (W4A16); each benchmark reports the precision it was built/run with.

## Directory Overview

```text
CuQwen/benchmark/
├── CMakeLists.txt         # Build configuration for the C++/CUDA benchmark
├── cuqwen_benchmark.cu    # Native CuQwen decode-throughput benchmark
├── kernel_profile.cu      # Per-kernel profiling harness
├── bench_vllm.py          # vLLM decode benchmark (FlashAttention + CUDA graphs)
└── bench_llamacpp.py      # llama.cpp decode benchmark (wraps native llama-bench)
```

## 1. CuQwen Benchmark

The native benchmark samples throughput at **9 context depths** (a timed 0.5K decode slice ending at each of `1K, 5K, 9K, 13K, 17K, 21K, 25K, 29K, 32K`); the initial `0–0.5K` slice is an untimed warmup. Decode work between sampled windows is skipped by advancing the position directly — since the KV cache is preallocated and attention cost scales with context length, per-token throughput at each sampled depth stays representative.

### Export Model Weights

From the project root, convert the Hugging Face weights into CuQwen's binary format. All three arguments are **required**:

```bash
# From project root
python3 export_weights.py --model=<family> --modelsize=<size> --quantization=<quant>
```

where `family` is `qwen2.5` or `qwen3`, `size` is one of that family's sizes
(`qwen2.5`: `0.5b`, `1.5b`, `3b`, `7b`, `14b` · `qwen3`: `0.6b`, `1.7b`, `4b`, `8b`, `14b`),
and `quant` is one of `fp16`, `int8`, `int4`.

### Build and Run

Build the benchmark executable from within the `benchmark/` folder. All four arguments are **required**, and `-Dmodel`/`-Dmodelsize`/`-Dquant` **must match** what was used during export:

```bash
cd benchmark
mkdir -p build && cd build
cmake -Dmodel=<family> -Dmodelsize=<size> -Dquant=<quant> -Dgpu_arch=<gpu_architecture> ..
make -j$(nproc)
cd ..

# Execute the CuQwen benchmark
./build/cuqwen_benchmark
```

`gpu_architecture` examples: `75` (RTX 2070), `86` (RTX 3090), `89` (RTX 4090), `90` (H100).

Example — Qwen3-8B in INT4 on an RTX 3090:

```bash
# from project root
python3 export_weights.py --model=qwen3 --modelsize=8b --quantization=int4
cd benchmark && mkdir -p build && cd build
cmake -Dmodel=qwen3 -Dmodelsize=8b -Dquant=int4 -Dgpu_arch=86 ..
make -j$(nproc) && cd ..
./build/cuqwen_benchmark
```

> **Note:** `include/config.h` caps `max_seq_len` at 8K for Qwen3 so the small models fit an
> 8 GB GPU. The benchmark samples depths up to 32K, so raise `max_seq_len` back to `32768`
> (and re-export/rebuild) before running the full-context Qwen3 benchmark on a larger GPU.

### Reported Metrics

* **Sampled Slice Throughput (tok/s):** decode speed of a 0.5K slice at each of the 9 sampled depths.
* **Average Speed (tok/s):** mean throughput across the 9 depths.
* **Speed Decay Rate (%):** drop in throughput from the first sampled depth (~1K) to the last (~32K):

$$\text{Decay Rate} = \frac{\text{Speed}_{1\text{K}} - \text{Speed}_{32\text{K}}}{\text{Speed}_{1\text{K}}} \times 100$$

---

## 2. vLLM & llama.cpp Benchmarks

`bench_vllm.py` and `bench_llamacpp.py` measure the same single-stream decode throughput on the two reference engines, across 9 context depths (`500, 3500, 7500, …, 31500`), decoding 500 tokens at each. Both scripts take `--model` (family), `--modelsize`, and `--quant`, all required; there are no default configurations, and results are printed to stdout. (These mirror CuQwen's own `-Dmodel`/`-Dmodelsize`/`-Dquant` flags.)

* **`bench_vllm.py`** — runs vLLM (FlashAttention + CUDA graphs); decode is timed from the engine's first→last generated-token interval, so prefill is excluded.
* **`bench_llamacpp.py`** — wraps the native `llama-bench` (CUDA graphs + FlashAttention, all layers on GPU), pulling the official Qwen GGUF weights.

The weights for each `--quant` are auto-selected from these repositories (`<S>` is the HF size tag, e.g. `3B` or `8B`):

**Qwen2.5** (`--model qwen2.5`, sizes `0.5b/1.5b/3b/7b/14b`):

| `--quant` | vLLM repo | llama.cpp GGUF (`Qwen/Qwen2.5-<S>-Instruct-GGUF`) |
|-----------|-----------|----------------|
| `fp16` (W16A16) | `Qwen/Qwen2.5-<S>-Instruct` | `…-fp16.gguf` |
| `int8` (W8A16) | `Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int8` | `…-q8_0.gguf` |
| `int4` (W4A16) | `Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int4` | `…-q4_0.gguf` |

**Qwen3** (`--model qwen3`, sizes `0.6b/1.7b/4b/8b/14b`) — Qwen's official quantizations differ from Qwen2.5, so the defaults differ too:

| `--quant` | vLLM repo | llama.cpp GGUF (`Qwen/Qwen3-<S>-GGUF`) |
|-----------|-----------|----------------|
| `fp16` (W16A16) | `Qwen/Qwen3-<S>` | *(no official full-precision GGUF — pass `--gguf PATH`)* |
| `int8` (W8A16) | *(no official W8A16 repo — pass `--repo PATH`)* | `…-Q8_0.gguf` |
| `int4` (W4A16) | `Qwen/Qwen3-<S>-AWQ` *(only `4b`/`8b`/`14b`)* | `…-Q4_K_M.gguf` |

Qwen3 caveats (why some cells need an override):
- **vLLM `int8`:** Qwen ships Qwen3 as FP8 (W8A8), not a weights-only W8A16 GPTQ build, so there is no like-for-like repo. Pass `--repo <hf_repo>` with a third-party GPTQ-Int8 checkpoint to benchmark it.
- **vLLM `int4`:** official 4-bit is AWQ (`Qwen/Qwen3-<S>-AWQ`), published only for `4b`/`8b`/`14b`; for `0.6b`/`1.7b` pass `--repo`.
- **llama.cpp:** the official GGUF repo has no `fp16` file and no legacy `q4_0`; the closest official 4-bit is `Q4_K_M`, and `0.6b`/`1.7b` ship only `Q8_0`. Pass `--gguf PATH` to benchmark any other GGUF (e.g. an `fp16`/`q4_0` build you convert yourself). GGUF names are matched case-insensitively.

Weights are downloaded on demand on first use.

### Requirements

- NVIDIA GPU with a recent driver and enough VRAM for the chosen model/precision (e.g. a 24 GB RTX 3090 handles 7B at FP16 with a 32K KV cache). Validated on an RTX 3090 (24 GB).
- Python 3.10+ and the CUDA toolkit (`nvcc`) for building llama.cpp.

### Install Dependencies

```bash
pip install --no-cache-dir vllm==0.26.0
pip install --no-cache-dir transformers huggingface_hub numpy

# vLLM 0.26 on some images ships a broken torchcodec that crashes the import.
# It is only used for video models, so remove it:
pip uninstall -y torchcodec

# If HuggingFace's Xet transfer backend is flaky, force plain HTTPS downloads:
export HF_HUB_DISABLE_XET=1
```

### Build llama.cpp (with CUDA)

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build \
      -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=86   # 86=RTX30xx/A10 · 89=RTX40xx · 80=A100 · 90=H100
cmake --build llama.cpp/build --target llama-bench -j
# -> binary at llama.cpp/build/bin/llama-bench (CUDA graphs + FlashAttention on by default)
```

`bench_llamacpp.py` finds the binary automatically at `llama.cpp/build/bin/llama-bench` (or set `LLAMA_BENCH=/path/to/llama-bench`, or pass `--llama-bench PATH`).

### Run

```bash
# vLLM — Qwen2.5-3B, 4-bit weights
python3 bench_vllm.py --model qwen2.5 --modelsize 3b --quant int4

# vLLM — Qwen3-8B, 4-bit (AWQ) weights
python3 bench_vllm.py --model qwen3 --modelsize 8b --quant int4

# llama.cpp — Qwen2.5-7B, fp16
python3 bench_llamacpp.py --model qwen2.5 --modelsize 7b --quant fp16

# llama.cpp — Qwen3-4B, 8-bit (Q8_0) weights
python3 bench_llamacpp.py --model qwen3 --modelsize 4b --quant int8

# vLLM — Qwen3-8B int8 needs a third-party GPTQ-Int8 repo (no official one):
python3 bench_vllm.py --model qwen3 --modelsize 8b --quant int8 --repo <hf_gptq_int8_repo>
```

### What is Measured

- **Decode only.** Prefill is excluded on both engines, so the number is steady-state tokens/s at that context depth, not end-to-end latency.
- **Batch = 1** (single stream) — the regime this suite targets; vLLM's continuous batching is not exercised here.
- **FP16 KV cache** on both. Weight-only quantization: GPTQ (vLLM) and Q4_0/Q8_0 (llama.cpp).
- `bench_llamacpp.py` measures the native `llama-bench` directly (CUDA graphs + FlashAttention, all layers on GPU) — llama.cpp's fastest single-stream decode path, for a fair engine-to-engine comparison.

---

## Comprehensive Benchmark Results

For the full per-depth data, averages, speedups, and decay analysis across all five model sizes (`0.5B`, `1.5B`, `3B`, `7B` in `FP16`/`INT8`/`INT4`, plus `14B` in `INT4`), refer to the main documentation:

👉 [**CuQwen 1.2 Benchmark Analysis (RELEASE_1.2_BENCHMARK.md)**](../docs/RELEASE_1.2_BENCHMARK.md)
