# CuQwen (v3_optimizedmemoryaccess) — CUDA C++ Qwen2.5 Inference Engine

`v3_optimizedmemoryaccess` is the third iteration of the **CuQwen** C++/CUDA inference engine for **Qwen2.5-1.5B-Instruct**. Building directly upon the monolithic kernel fusion introduced in `v2_kernelfusion`, this release optimizes VRAM throughput by converting scalar DRAM accesses into vectorized 128-bit memory instructions (`float4` / `half2`) and restructuring thread-block tiling patterns to achieve coalesced global memory transactions.

---

## Technical Overview

* **128-Bit Vectorized Memory Access:** Replaces unvectorized scalar loads and stores across all custom kernels with vectorized CUDA types (`float4` and `half2`), maximizing memory bandwidth utilization per instruction cycle.
* **Coalesced Global Memory Transactions:** Restructures thread-block indexing and warp layout to ensure aligned, contiguous 32-byte memory transactions when loading matrix weights and activations from VRAM.
* **Zero External Dependencies:** Built purely in C++/CUDA runtime without reliance on PyTorch, LibTorch, ONNX, or Python execution environments.

> **Configuration Tip:** To adjust the maximum sequence capacity for longer context windows, modify the `max_seq_len` parameter in `include/config.h` before compiling:
> ```cpp
> static constexpr int32_t max_seq_len = 8192; // Adjust maximum sequence context length
> ```

---

## Directory Structure

```text
cuda/v3_optimizedmemoryaccess/
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
`v3_optimizedmemoryaccess` provides built-in benchmarking utilities to measure raw CUDA generation throughput against PyTorch/Hugging Face baseline implementations.

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