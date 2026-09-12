# CuQwen (v2_kernelfusion) — CUDA C++ Qwen2.5 Inference Engine

`v2_kernelfusion` is an experimental iteration of the **CuQwen** C++/CUDA inference engine for **Qwen2.5-1.5B-Instruct**. This release completely eliminates cuBLAS calls from the core execution loop, replacing standard BLAS routines with four hand-written fused CUDA kernels per Transformer layer to keep intermediate activations inside registers and shared memory.

---

## Technical Overview

* **Complete cuBLAS Elimination:** Replaces discrete `cublasGemmEx` matrix-vector calls with end-to-end custom CUDA kernel implementations across all 28 Transformer layers.
* **4-Stage Fused Pipeline:** Reorganizes the per-layer forward pass into four monolithic custom kernels:
  1. **Fused Attention Input:** Combines FP16 Input RMSNorm, Q/K/V weight projections ($W_q, W_k, W_v$), bias addition, and RoPE rotary positional encoding in a single pass without writing unnormalized vectors to VRAM.
  2. **Fused Attention Output:** Fuses the output projection ($W_o$) dot-product accumulation with the first skip-connection residual addition.
  3. **Fused MLP Stage 1:** Unifies Post-Attention RMSNorm, dual gate ($W_{\text{gate}}$) and up ($W_{\text{up}}$) projections, and SwiGLU activation.
  4. **Fused MLP Stage 2:** Integrates down projection ($W_{\text{down}}$) and the second skip-connection residual addition.
* **Kernel Launch Reduction:** Drops per-layer GPU kernel dispatches from 17 down to 5 for tranformer layers, eliminating CPU scheduling overhead.
* **Zero External Dependencies:** Built purely in C++/CUDA runtime without reliance on PyTorch, LibTorch, ONNX, or Python execution environments.

> **Configuration Tip:** To adjust the maximum sequence capacity for longer context windows, modify the `max_seq_len` parameter in `include/config.h` before compiling:
> ```cpp
> static constexpr int32_t max_seq_len = 8192; // Adjust maximum sequence context length
> ```

---

## Directory Structure

```text
cuda/v2_kernelfusion/
├── CMakeLists.txt          # CMake build pipeline
├── export_weights.py       # Converts HuggingFace Safetensors to packed binary format
├── include/                # Header files
│   ├── config.h
│   ├── kernels.cuh
│   ├── load_weights.h
│   ├── model.cuh
│   └── tokenizer.h
├── src/                    # Implementation files
│   ├── kernels.cu
│   ├── load_weights.cpp
│   ├── main.cu
│   ├── model.cu
│   └── tokenizer.cpp
└── benchmark/              # Comparative benchmarking suite
    ├── CMakeLists.txt
    ├── cuqwen_benchmark.cu
    └── hugingface_benchmark.py
```

## Build & Usage Guide

### Prerequisites
* Linux OS (Ubuntu 22.04/24.04 recommended)
* NVIDIA GPU with CUDA Toolkit installed (v12.0+)
* gcc/g++ compiler supporting C++17
* CMake 3.18+
* Python 3.8+

### Install System & Dependencies
```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install build-essential cmake pkg-config libpcre2-dev nlohmann-json3-dev python3-pip python3-dev -y
pip3 install --no-cache-dir --break-system-packages huggingface_hub numpy
```

## Step 1: Export Weights & Tokenizer
Run the export script to download Qwen/Qwen2.5-1.5B-Instruct from Hugging Face and serialize the FP16 weights into a single contiguous binary file (weights/model_fp16_1_5b.bin):

```bash
python3 export_weights.py
```

## Step 2: Compile & Run the Interactive Chat Engine
Build the bare-metal C++/CUDA chat binary using CMake:

```bash
mkdir -p build && cd build

# Replace '75' with your target GPU compute capability
cmake -Dgpu_arch=75 ..
make -j$(nproc)
cd ..

# Run interactive inference engine
./build/cuqwen
```

## Step 3: Running Benchmarks
`v2_kernelfusion` provides built-in benchmarking utilities to measure raw CUDA generation throughput against PyTorch/Hugging Face baseline implementations.

### 1. CUDA Benchmark
To measure end-to-end token generation performance of the C++/CUDA pipeline. Make sure to export weights and tokenizer before running the benchmark:

```bash
python3 export_weights.py

cd benchmark
mkdir -p build && cd build

# Replace '75' with your target GPU compute capability
cmake -Dgpu_arch=75 ..
make -j$(nproc)
cd ..

# Run CUDA benchmark executable
./build/cuqwen_benchmark
```

### 2. Hugging Face PyTorch Benchmark
To run the PyTorch native baseline benchmark using Hugging Face Transformers (requires torch and transformers installed in your environment):

```bash
cd benchmark
python3 hugingface_benchmark.py
```