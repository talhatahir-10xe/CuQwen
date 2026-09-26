# CuQwen: High-Performance CUDA Inference Engine for Qwen

![CuQwen](/assets/CuQwen.jpeg)

***Note: This is the current working branch for the next release in which I'm working on supporting int8 and int4 weights only quantization for the current models in CuQwen (W8A16 and W4A16)***

CuQwen is a C++/CUDA inference engine written purely from scratch to explore the theoretical performance limits of single-user (```Batch Size 1```) autoregressive token generation using optimized custom CUDA kernels for open source Qwen models that can run on local edge devices for Nvidia GPUs.

## Performance
Average inference speed (tokens/second) for **Qwen2.5 Instruct model** across 32K context window on RTX 3090 

| Model Size | CuQwen | vLLM | Ollama |
| ---------- | ------ | ---- | ------ |
| 0.5b       |   462  | 398  | 355    |
| 1.5b       |   203  | 172  | 139    |
| 3b         |   113  | 101  | 106    |
| 7b         |   55   | 48   | 54     |

Complete benchmarking results and details can be found [here](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_BENCHMARK.md).

## Supported Models

CuQwen natively supports the **Qwen2.5** model family across multiple parameter scales, in `fp16` or weights-only `int8` (W8A16) / `int4` (W4A16) precision:

* **Qwen2.5-0.5B** (`0.5b`)
* **Qwen2.5-1.5B** (`1.5b`)
* **Qwen2.5-3B** (`3b`)
* **Qwen2.5-7B** (`7b`)

Quantizied models and latest Qwen model series (`Qwen 3.0`, `3.5`, `3.6`, `3.7` and `3.8`) will also be supported in next releases 

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
python3 export_weights.py --model=<model_size> --quantization=<quant>
```
where, `model_size` can be `0.5b`, `1.5b`, `3b` or `7b`, and `quant` can be `fp16` (default), `int8` or `int4`.

`--quantization=int8` / `int4` enables **weights-only quantization (W8A16 / W4A16)**: the linear-projection weights (`q/k/v/o_proj`, `gate/up/down_proj`) are stored quantized with per-group FP16 scales (128 weights per scale) and dequantized to FP16 in-kernel; `int4` additionally packs two weights per byte. Token embeddings / LM head, RMSNorm weights, biases, and the KV cache remain FP16. This shrinks the projection-weight footprint to roughly 1/2 (`int8`) or 1/4 (`int4`), e.g. Qwen2.5-3B fits comfortably in 8 GB VRAM where the FP16 model would not.

2. **Build the Engine**: Specify the model size, target GPU compute architecture (e.g., `75` for RTX 2070, `86` for RTX 3090, `89` for RTX 4090), and — matching the exported weights — the precision.

```bash
mkdir -p build
cd build
cmake -Dmodel=<model_size> -Dgpu_arch=<gpu_architecture> -Dquant=<quant> ..
make -j$(nproc)
cd ..
```
where `quant` is `fp16` (default), `int8` or `int4`, and **must match the `--quantization` used during export** (the engine validates this against the binary header). Omitting `-Dquant` builds the FP16 engine.
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
The `benchmark/` directory also has the scripts for `vLLM` and `Ollama` which were used to benchmark those inference engines. To run these benchmarking scripts please reffer [here](https://github.com/talhatahir-10xe/CuQwen/blob/main/benchmark/README.md) 

## Roadmap & Future Work
The following enhancements are planned for future development cycles:

* **Optimized Prefill Kernels:** Implement parallel prefill kernels to process prompt tokens in parallel rather than using the single-token autoregressive decode path for prefill stage.
* **Weight Quantization:** Add low-precision INT8 and INT4 (W8A16/W4A16) quantization.
* **KV-Cache Quantization:** Support INT8/FP8 KV-cache quantization to cut attention memory bandwidth usage during long-context generation.
* **Modern Microarchitecture Tuning:** Extend custom kernel implementations for NVIDIA Hopper (`sm_90`) and Blackwell (`sm_100`) architectures utilizing `TMA` (Tensor Memory Accelerator), `DSMEM` (Distributed Shared memory) and `DPX` instructions.
* **Long-Context Throughput Optimization:** Further mitigate throughput decay across extended sequence lengths. ✅ *Done in release 1.1 — throughput decay across the 32K context window was cut roughly in half; see the [Release 1.1 Optimization](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_OPTIMIZATION.md) document.*
* **Support for Newer Qwen Architecture Series:** Add native kernel and engine support for newer iterations in the Qwen family, including Qwen 3.0, Qwen 3.5, and Qwen 3.8 models.

## Documentation

* [**Release 1.0 Benchmark**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.0_BENCHMARK.md): Benchmarking details and results of CuQwen 1.0 vs vLLM and Ollama
* [**Release 1.0 Optimization Journey**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.0_OPTIMIZATION_JOURNEY.md): The complete journey of the different optimizations used in the process of implementing CuQwen 1.0
* [**Release 1.1 Benchmark**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_BENCHMARK.md): Benchmarking details and results of CuQwen 1.1 vs vLLM and Ollama across the extended 32K context window
* [**Release 1.1 Optimization**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_OPTIMIZATION.md): The long-context optimizations applied in CuQwen 1.1 to reduce throughput decay over CuQwen 1.0
