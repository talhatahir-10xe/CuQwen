# CuQwen: High-Performance C++/CUDA Engine for Qwen

![CuQwen](/assets/CuQwen.jpeg)

CuQwen is a C++/CUDA inference engine written purely from scratch to explore the theoretical performance limits of single-user (```Batch Size 1```) autoregressive token generation using optimized custom CUDA kernels for open source Qwen models that can run on local edge devices.

## Performance
Average inference speed for **Qwen2.5 Instruct model** across 8K context window

| Model SIze | CuQwen | vLLM | Ollama |
| ---------- | ------ | ---- | ------ |
| 0.5b       |   490  | 471  | 371    |
| 1.5b       |   212  | 191  | 160    |
| 3b         |   114  | 109  | 112    |
| 7b         |   57   | 50   | 55     |

Complete benchmarking results and details can be found here.

## Supported Models

CuQwen natively supports the **Qwen2.5** model family across multiple parameter scales (`fp16` precision weights):

* **Qwen2.5-0.5B** (`0.5b`)
* **Qwen2.5-1.5B** (`1.5b`)
* **Qwen2.5-3B** (`3b`)
* **Qwen2.5-7B** (`7b`)

Quantizied models and latest Qwen model series will also be supported in next releases 

## Project Structure

```text
CuQwen/
├── assets/
├── benchmark/            # Head-to-head benchmarking vs. production engines (ollama and vllm)
├── CMakeLists.txt        # Top-level C++ build configuration
├── cuda_optimizations/   # Incremental CuQwen optimization journey
├── Dockerfile            # Container definition for reproducible environments
├── docs/                 # Detailed documentation on benchmarking and optimizations used in CuQwen
├── export_weights.py     # Python script to convert HuggingFace weights to CuQwen binary format
├── include/              # CuQwen headers and kernel declarations
├── README.md             # Project overview and setup guide
├── setup.sh              # Automated Docker environment initialization script
└── src/                  # Core CUDA kernels and runtime engine source files
```

## Getting Started

### Prerequisites
* Linux OS (Ubuntu 22.04/24.04 recommended)
* NVIDIA GPU with CUDA Toolkit installed (v12.0+)
* gcc/g++ compiler supporting C++17
* CMake 3.18+
* Python 3.8+

### Install Dependencies (Host Machine)
```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install build-essential cmake pkg-config libpcre2-dev nlohmann-json3-dev python3-pip python3-dev -y
pip3 install --no-cache-dir --break-system-packages huggingface_hub numpy
```
---

### Docker Setup (Alternative)

If you prefer a pre-configured environment with all dependencies and the CUDA Toolkit installed, run the automated setup script to build and launch the Docker container:

```bash
chmod +x setup.sh
./setup.sh
```

This builds the image, starts the container, and drops you into a shell with full GPU passthrough configured.

## Building and Running Interactive Chat

1. **Export Model Weights**: Convert Hugging Face weights to binary format for the target model_size (`0.5b`, `1.5b`, `3b`, or `7b`).

```bash
python3 export_weights.py --model=<model_size>
```
where, `model_size` can be `0.5b`, `1.5b`, `3b` or `7b`  

2. **Build the Engine**: Specify the model size and target GPU compute architecture (e.g., `75` for RTX 2070, `86` for RTX 3090, `89` for RTX 4090).

```bash
mkdir -p build
cd build
cmake -Dmodel=<model_size> -Dgpu_arch=<gpu_architecture> ..
make -j$(nproc)
cd ..
```
3. **Launch Interactive Chat**
```bash
./build/cuqwen
```

## Running Benchmarks

To run the internal benchmark suite measuring token generation latency across different context windows:

1. **Export Model Weights:**
```bash
python3 export_weights.py --model=<model_size>
```
2. **Build and Run Benchmark Binary:**
```bash
cd benchmark
mkdir -p build
cd build
cmake -Dmodel=<model_size> -Dgpu_arch=<gpu_architecture> ..
make -j$(nproc)
cd ..
./build/cuqwen_benchmark
```
The `benchmark/` directory also has the scripts for `vLLM` and `Ollama` which were used to benchmark those inference engines.

## Roadmap & Future Work
The following enhancements are planned for future development cycles:

* **Optimized Prefill Kernels:** Implement parallel prefill kernels to process prompt tokens in parallel rather than using the single-token autoregressive decode path for prefill stage.
* **Weight Quantization:** Add low-precision INT8 and INT4 (W8A16/W4A16) quantization.
* **KV-Cache Quantization:** Support INT8/FP8 KV-cache quantization to cut attention memory bandwidth usage during long-context generation.
* **Modern Microarchitecture Tuning:** Extend custom kernel implementations for NVIDIA Hopper (`sm_90`) and Blackwell (`sm_100`) architectures utilizing `TMA` (Tensor Memory Accelerator), `DSMEM` (Distributed Shared memory) and `DPX` instructions.
* **Long-Context Throughput Optimization:** Further mitigate throughput decay across extended sequence lengths.
* **Support for Newer Qwen Architecture Series:** Add native kernel and engine support for newer iterations in the Qwen family, including Qwen 3.0, Qwen 3.5, and Qwen 3.8 models.

## Documentation

* Benchmarking CuQwen: Contains the benchmarking details and results of CuQwen vs vLLM and Ollama
* Optimization journey: Contains the complete journey of different optimizations that were used in the process of implementing CuQwen