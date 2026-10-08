# CuQwen Benchmarking Suite

This directory contains the benchmarker applications and evaluation scripts used to compare **CuQwen** against the **vLLM** production inference framework.

The benchmark measures bare-metal, single-user (`Batch Size 1`) autoregressive decode speed across a **32k context window** sampled in **1k slice increments**.

## Directory Overview

```text
CuQwen/benchmark/
├── CMakeLists.txt         # Build configuration for C++/CUDA benchmarks
├── cuqwen_benchmark.cu    # Native CUDA benchmark binary source
├── hf_benchmark.py        # Baseline Hugging Face Transformers benchmark script
└── vllm_benchmark.py      # AsyncLLMEngine benchmark script for vLLM (FP16)
```

## Benchmark Prerequisites & Environment Setup

All benchmarking requires an NVIDIA GPU with sufficient VRAM (e.g., 20GB VRAM card for 7B models at full FP16 precision with an 32k KV cache).

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

## Benchmark Output Metrics

Both framework runners record and report identical metric parameters to allow direct head-to-head comparisons:

* **Token Slice Throughput (tok/s):** Measured speed across each 1,000-token context chunk ($0\rightarrow1\text{k}, 1\text{k}\rightarrow2\text{k}, \dots, 7\text{k}\rightarrow32\text{k}$).
* **Average Speed (tok/s):** Mean generation speed across the entire 32k context window.
* **Speed Decay Rate (%):** Percentage drop in throughput from the initial 1k slice to the final 32k slice:

$$\text{Decay Rate} = \frac{\text{Speed}_{1\text{k}} - \text{Speed}_{8\text{k}}}{\text{Speed}_{1\text{k}}} \times 100$$

---

## Comprehensive Benchmark Results

For full chart breakdowns, architectural visualizer comparisons, and trade-off analysis across all four model sizes (`0.5B`, `1.5B`, `3B`, `7B`), refer to the main documentation:

👉 [**CuQwen Benchmark Strategy & Analysis (RELEASE_1.1_BENCHMARK.md)**](../docs/RELEASE_1.1_BENCHMARK.md)
