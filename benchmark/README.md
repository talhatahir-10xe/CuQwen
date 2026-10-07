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

From the project root, convert the Hugging Face weights into CuQwen's binary format. Both arguments are **required**:

```bash
# From project root
python3 export_weights.py --model=<model_size> --quantization=<quant>
```

where `model_size` is one of `0.5b`, `1.5b`, `3b`, `7b`, and `quant` is one of `fp16`, `int8`, `int4`.

### Build and Run

Build the benchmark executable from within the `benchmark/` folder. All three arguments are **required**, and `-Dquant` **must match** the precision used during export:

```bash
cd benchmark
mkdir -p build && cd build
cmake -Dmodel=<model_size> -Dquant=<quant> -Dgpu_arch=<gpu_architecture> ..
make -j$(nproc)
cd ..

# Execute the CuQwen benchmark
./build/cuqwen_benchmark
```

`gpu_architecture` examples: `75` (RTX 2070), `86` (RTX 3090), `89` (RTX 4090), `90` (H100).

### Reported Metrics

* **Sampled Slice Throughput (tok/s):** decode speed of a 0.5K slice at each of the 9 sampled depths.
* **Average Speed (tok/s):** mean throughput across the 9 depths.
* **Speed Decay Rate (%):** drop in throughput from the first sampled depth (~1K) to the last (~32K):

$$\text{Decay Rate} = \frac{\text{Speed}_{1\text{K}} - \text{Speed}_{32\text{K}}}{\text{Speed}_{1\text{K}}} \times 100$$

---

## 2. vLLM & llama.cpp Benchmarks

`bench_vllm.py` and `bench_llamacpp.py` measure the same single-stream decode throughput on the two reference engines, across 9 context depths (`500, 3500, 7500, …, 31500`), decoding 500 tokens at each. Both scripts take `--model` and `--quant` and require them; there are no default configurations, and results are printed to stdout.

* **`bench_vllm.py`** — runs vLLM (FlashAttention + CUDA graphs); decode is timed from the engine's first→last generated-token interval, so prefill is excluded.
* **`bench_llamacpp.py`** — wraps the native `llama-bench` (CUDA graphs + FlashAttention, all layers on GPU), pulling the official Qwen GGUF weights.

The weights for each `--quant` come from these repositories:

| `--quant` | vLLM repo | llama.cpp GGUF |
|-----------|-----------|----------------|
| `fp16` (W16A16) | `Qwen/Qwen2.5-<S>-Instruct` | `…-fp16.gguf` |
| `int8` (W8A16) | `Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int8` | `…-q8_0.gguf` |
| `int4` (W4A16) | `Qwen/Qwen2.5-<S>-Instruct-GPTQ-Int4` | `…-q4_0.gguf` |

where `<S>` is the model size (`0.5B`, `1.5B`, `3B`, `7B`). Weights are downloaded on demand on first use.

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
python3 bench_vllm.py --model 3b --quant int4

# llama.cpp — Qwen2.5-7B, fp16
python3 bench_llamacpp.py --model 7b --quant fp16
```

### What is Measured

- **Decode only.** Prefill is excluded on both engines, so the number is steady-state tokens/s at that context depth, not end-to-end latency.
- **Batch = 1** (single stream) — the regime this suite targets; vLLM's continuous batching is not exercised here.
- **FP16 KV cache** on both. Weight-only quantization: GPTQ (vLLM) and Q4_0/Q8_0 (llama.cpp).
- `bench_llamacpp.py` measures the native `llama-bench` directly. Running the same GGUF through the **Ollama** server was ~2× slower (serving-layer overhead), so llama.cpp is benchmarked directly for a fair engine comparison.

---

## Comprehensive Benchmark Results

For the full per-depth data, averages, speedups, and decay analysis across all four model sizes (`0.5B`, `1.5B`, `3B`, `7B`) and all three precisions (`FP16`, `INT8`, `INT4`), refer to the main documentation:

👉 [**CuQwen 1.2 Benchmark Analysis (RELEASE_1.2_BENCHMARK.md)**](../docs/RELEASE_1.2_BENCHMARK.md)
