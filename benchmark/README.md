# CuQwen Benchmarking Suite

This directory contains the benchmarker applications and evaluation scripts used to compare **CuQwen** against production LLM inference frameworks (**vLLM** and **Ollama**).

The benchmark measures bare-metal, single-user (`Batch Size 1`) autoregressive decode speed across an **8k context window** sampled in **1k slice increments**.

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

All benchmarking requires an NVIDIA GPU with sufficient VRAM (e.g., NVIDIA RTX 3090/4090 for 7B models at full FP16 precision with an 8k KV cache).

### 1. CuQwen Benchmark Setup

#### Export Model Weights

First, convert the Hugging Face weights into CuQwen's raw binary format from the root directory:

```bash
# From project root
python3 export_weights.py --model=<model_size>
```

#### Build and Run Binary

Build the native C++/CUDA benchmark executable from within the `benchmark/` folder:

```bash
cd benchmark
mkdir -p build && cd build
cmake -Dmodel=<model_size> -Dgpu_arch=<gpu_architecture> ..
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

All three framework runners record and report identical metric parameters to allow direct head-to-head comparisons:

* **Token Slice Throughput (tok/s):** Measured speed across each 1,000-token context chunk ($0\rightarrow1\text{k}, 1\text{k}\rightarrow2\text{k}, \dots, 7\text{k}\rightarrow8\text{k}$).
* **Average Speed (tok/s):** Mean generation speed across the entire 8k context window.
* **Speed Decay Rate (%):** Percentage drop in throughput from the initial 1k slice to the final 8k slice:

$$\text{Decay Rate} = \frac{\text{Speed}_{1\text{k}} - \text{Speed}_{8\text{k}}}{\text{Speed}_{1\text{k}}} \times 100$$

---

## Comprehensive Benchmark Results

For full chart breakdowns, architectural visualizer comparisons, and trade-off analysis across all four model sizes (`0.5B`, `1.5B`, `3B`, `7B`), refer to the main documentation:

👉 [**CuQwen Benchmark Strategy & Analysis (BENCHMARK.md)**](../docs/BENCHMARK.md)
