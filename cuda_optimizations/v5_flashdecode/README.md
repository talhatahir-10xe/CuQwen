# CuQwen (v5_flashdecode) — CUDA C++ Qwen2.5 Inference Engine

`v5_flashdecode` is the fifth iteration of the **CuQwen** C++/CUDA inference engine for **Qwen2.5-1.5B-Instruct**. Building upon the kernel fusion (`v2`), 128-bit memory vectorization (`v3`), and warp-level primitives (`v4`), this release replaces standard sequential attention during generation with a custom **FlashDecode** kernel architecture to maintain flat latency scaling even across extended sequence context lengths.

---

## Technical Overview

* **Parallel KV-Cache Partitioning (FlashDecode):** Splits the Key/Value sequence length dimension into independent fixed blocks (```SPLIT_K_SIZE = 256```), enabling maximum Streaming Multiprocessor (SM) occupancy on single-batch, long-context decoding steps.
* **Two-Phase Flash Attention Decoding:**
  1. **Partial Softmax Generation:** Computes local tile dot-product query-key attention ($Q \cdot K^T$), tracks partial block maximums and exponential sums, and accumulates weighted partial output vectors into FP32 temporary work buffers (```d_partial_out```, ```d_partial_max```, ```d_partial_sum```).
  2. **Cross-Split Rescaling & Reduction:** A dedicated reduction kernel merges partial block outputs across sequence splits by calculating global maximums and normalized online exponential weights to write final FP16 exact attention outputs into memory.
* **Zero External Dependencies:** Built purely in C++/CUDA runtime without reliance on PyTorch, LibTorch, ONNX, FlashAttention-v2 C++ bindings, or Python execution environments.

> **Configuration Tip:** To adjust the maximum sequence capacity for longer context windows, modify the `max_seq_len` parameter in `include/config.h` before compiling:
> ```cpp
> static constexpr int32_t max_seq_len = 8192; // Adjust maximum sequence context length
> ```

---

## Directory Structure

```text
cuda/v5_flashdecode/
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
`v5_flashdecode` provides built-in benchmarking utilities to measure raw CUDA generation throughput against PyTorch/Hugging Face baseline implementations.

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