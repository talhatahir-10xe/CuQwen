# CuQwen Benchmarking Suite

This directory contains the benchmarker applications and evaluation scripts used to compare **CuQwen** against production LLM inference frameworks (**vLLM** and **Ollama**).

The benchmark measures bare-metal, single-user (`Batch Size 1`) autoregressive decode speed across the **32k context window**. To keep runs fast, throughput is **sampled at 9 context depths** (a timed 0.5K decode slice ending at each of `1K, 5K, 9K, 13K, 17K, 21K, 25K, 29K, 32K`) rather than decoding every token; the initial `0–0.5K` slice is an untimed warmup. Decode work between sampled windows is skipped by advancing the position directly — since the KV cache is preallocated and attention cost scales with context length, per-token throughput at each sampled depth stays representative.

CuQwen supports `fp16`, `int8` (W8A16) and `int4` (W4A16) weights; the benchmark reports the precision it was built with.

## Directory Overview

```text
CuQwen/benchmark/
├── CMakeLists.txt         # Build configuration for C++/CUDA benchmarks
├── cuqwen_benchmark.cu    # Native CUDA benchmark binary source
├── hf_benchmark.py        # Baseline Hugging Face Transformers benchmark script
├── ollama_benchmark.py    # Async streaming benchmark script for Ollama (FP16)
└── vllm_benchmark.py      # AsyncLLMEngine benchmark script for vLLM (FP16)
```

## Benchmark Prerequisites & Environment Setup

All benchmarking requires an NVIDIA GPU with sufficient VRAM (e.g., 20GB VRAM card for 7B models at full FP16 precision with an 32k KV cache).

### 1. CuQwen Benchmark Setup

#### Export Model Weights

First, convert the Hugging Face weights into CuQwen's raw binary format from the root directory:

```bash
# From project root
python3 export_weights.py --model=<model_size> --quantization=<quant>
```

where `quant` is `fp16` (default), `int8` or `int4`.

#### Build and Run Binary

Build the native C++/CUDA benchmark executable from within the `benchmark/` folder. The `-Dquant` value **must match** the precision used during export (default `fp16`):

```bash
cd benchmark
mkdir -p build && cd build
cmake -Dmodel=<model_size> -Dgpu_arch=<gpu_architecture> -Dquant=<quant> ..
make -j$(nproc)
cd ..

# Execute CuQwen benchmark
./build/cuqwen_benchmark
```

`gpu_architecture` examples: `75` (RTX 2070), `86` (RTX 3090), `89` (RTX 4090)

### 2. vLLM Benchmark Setup

Install vLLM (`0.26.0`) and run the `vllm_benchmark.py` script:

```bash
# Install vLLM
pip3 install --no-cache-dir --break-system-packages vllm==0.26.0

# Run vLLM benchmark across 8k context window
python3 vllm_benchmark.py --model=<model_size>
```

### 3. Ollama Benchmark Setup

Install Ollama, launch the background daemon, prepare FP16 GGUF model files, and run the benchmark script:

#### Install and Start Ollama Engine

```bash
apt-get update && apt-get install -y curl zstd
curl -fsSL https://ollama.com/install.sh | sh

# Configure environment for pure single-sequence testing
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1

# Launch server in background
ollama serve > /dev/null 2>&1 &

# Install Python dependencies
pip3 install ollama numpy huggingface_hub --break-system-packages
```

#### Prepare Model & Run Benchmark (Example for 1.5B)

```bash
# Download target FP16 GGUF model
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf --local-dir .

# Create local Ollama model manifest
echo "FROM ./qwen2.5-1.5b-instruct-fp16.gguf" > Modelfile
ollama create qwen2.5:1.5b-fp16 -f Modelfile

# Run Ollama benchmark
python3 ollama_benchmark.py --model=1.5b
```

## Benchmark Output Metrics

All framework runners record and report the same metric parameters to allow direct head-to-head comparisons:

* **Sampled Slice Throughput (tok/s):** Measured speed of a 0.5K decode slice at each of the 9 sampled context depths ($1\text{k}, 5\text{k}, 9\text{k}, \dots, 29\text{k}, 32\text{k}$).
* **Average Speed (tok/s):** Mean throughput across the 9 sampled depths.
* **Speed Decay Rate (%):** Percentage drop in throughput from the first sampled depth (~1k) to the last (~32k):

$$\text{Decay Rate} = \frac{\text{Speed}_{1\text{k}} - \text{Speed}_{32\text{k}}}{\text{Speed}_{1\text{k}}} \times 100$$

---

## Comprehensive Benchmark Results

For full chart breakdowns, architectural visualizer comparisons, and trade-off analysis across all four model sizes (`0.5B`, `1.5B`, `3B`, `7B`), refer to the main documentation:

👉 [**CuQwen Benchmark Strategy & Analysis (RELEASE_1.1_BENCHMARK.md)**](../docs/RELEASE_1.1_BENCHMARK.md)
