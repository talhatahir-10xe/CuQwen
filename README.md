# CuQwen: High-Performance CUDA Inference Engine for Qwen

![CuQwen](/assets/CuQwen.jpeg)

CuQwen is a C++/CUDA inference engine written purely from scratch to explore the theoretical performance limits of single-user (```Batch Size 1```) autoregressive token generation using optimized custom CUDA kernels for open source Qwen models that can run on local edge devices for Nvidia GPUs.

## Performance
Average decode speed (tokens/second) for the **Qwen2.5 Instruct** models across the 32K context window on an RTX 3090, in each supported weights-only precision. CuQwen posts the highest average throughput in every configuration that fits a 24 GB GPU — and stays level with the best (llama.cpp) on the heaviest one, Qwen2.5-14B INT4.

Each engine column is split into the three weights-only precisions (**FP16** W16A16, **INT8** W8A16, **INT4** W4A16). `✗` marks a configuration that does not fit a 24 GB RTX 3090 with the full 32K FP16 KV cache (see the note under the table).

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Model</th>
      <th colspan="3" align="center">CuQwen</th>
      <th colspan="3" align="center">vLLM</th>
      <th colspan="3" align="center">llama.cpp</th>
    </tr>
    <tr>
      <th align="center">FP16</th><th align="center">INT8</th><th align="center">INT4</th>
      <th align="center">FP16</th><th align="center">INT8</th><th align="center">INT4</th>
      <th align="center">FP16</th><th align="center">INT8</th><th align="center">INT4</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>0.5b</td><td align="center"><b>506</b></td><td align="center"><b>603</b></td><td align="center"><b>664</b></td><td align="center">403</td><td align="center">432</td><td align="center">478</td><td align="center">387</td><td align="center">490</td><td align="center">536</td></tr>
    <tr><td>1.5b</td><td align="center"><b>209</b></td><td align="center"><b>296</b></td><td align="center"><b>364</b></td><td align="center">171</td><td align="center">236</td><td align="center">286</td><td align="center">186</td><td align="center">265</td><td align="center">340</td></tr>
    <tr><td>3b</td><td align="center"><b>115</b></td><td align="center"><b>176</b></td><td align="center"><b>236</b></td><td align="center">100</td><td align="center">148</td><td align="center">193</td><td align="center">108</td><td align="center">165</td><td align="center">225</td></tr>
    <tr><td>7b</td><td align="center"><b>55</b></td><td align="center"><b>95</b></td><td align="center"><b>144</b></td><td align="center">48</td><td align="center">85</td><td align="center">125</td><td align="center">53</td><td align="center">91</td><td align="center">138</td></tr>
    <tr><td>14b</td><td align="center">✗</td><td align="center">✗</td><td align="center"><b>66</b></td><td align="center">✗</td><td align="center">✗</td><td align="center">62</td><td align="center">✗</td><td align="center">✗</td><td align="center">66</td></tr>
  </tbody>
</table>

> **14B — INT4 only on a 24 GB GPU.** The RTX 3090 has **24 GB** of VRAM, which cannot hold Qwen2.5-14B in **FP16** (~28 GB of weights) or **INT8** (~15 GB of weights **+** ~13 GB for the 32K FP16 KV cache ≈ 28 GB) — so those cells are marked ✗ for every engine; only **INT4** (≈9 GB weights + 13 GB KV) fits the full 32K window.

Complete benchmarking results and details can be found [here](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.2_BENCHMARK.md).

> The table above is the Qwen2.5 release-1.2 benchmark on the RTX 3090. **Qwen3** support landed in release 1.3; its benchmark numbers will be published separately.

## Supported Models

CuQwen natively supports the **Qwen2.5** and **Qwen3** dense model families across multiple
parameter scales, in `fp16` or weights-only `int8` (W8A16) / `int4` (W4A16) precision. The
engine is selected at build time with two flags: `-Dmodel=<family>` and `-Dmodelsize=<size>`.

**Qwen2.5** (`-Dmodel=qwen2.5`):

* **Qwen2.5-0.5B** (`0.5b`)
* **Qwen2.5-1.5B** (`1.5b`)
* **Qwen2.5-3B** (`3b`)
* **Qwen2.5-7B** (`7b`)
* **Qwen2.5-14B** (`14b`) — fits a 24 GB GPU in `int4` with the full 32K FP16 KV cache

**Qwen3** (`-Dmodel=qwen3`) — *new in release 1.3*:

* **Qwen3-0.6B** (`0.6b`)
* **Qwen3-1.7B** (`1.7b`)
* **Qwen3-4B** (`4b`)
* **Qwen3-8B** (`8b`)
* **Qwen3-14B** (`14b`)

Qwen3 adds per-head **QK-RMSNorm**, drops the QKV projection bias, and uses a decoupled
`head_dim` of 128 (so the attention width is independent of the hidden size); the KV cache
stays FP16. The engine ships with the context window capped at **8K** so the smaller Qwen3
models fit an 8 GB GPU (e.g. RTX 2070) — raise `max_seq_len` in `include/config.h` for a
larger GPU. Qwen3 `0.6b`/`1.7b`/`4b` have tied embeddings; `8b`/`14b` use a separate
(quantizable) LM head.

Later Qwen series (`Qwen 3.5`, `Qwen 3.8`) will be supported in future releases.

## Project Structure

```text
CuQwen/
├── assets/
├── benchmark/            # Head-to-head benchmarking vs. production engines (llama.cpp and vLLM)
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

1. **Export Model Weights**: Convert Hugging Face weights to CuQwen's binary format for the target family and size.

```bash
python3 export_weights.py --model=<family> --modelsize=<size> --quantization=<quant>
```
where `family` is `qwen2.5` or `qwen3`, `size` is one of the sizes for that family
(`qwen2.5`: `0.5b`, `1.5b`, `3b`, `7b`, `14b` · `qwen3`: `0.6b`, `1.7b`, `4b`, `8b`, `14b`),
and `quant` is one of `fp16`, `int8`, `int4`. All three arguments are **required**. The
output is written to `weights/model_<family>_<quant>_<size>.bin` plus a tokenizer JSON
(`weights/qwen25_bpe_ranks.json` or `weights/qwen3_bpe_ranks.json`).

`--quantization=int8` / `int4` enables **weights-only quantization (W8A16 / W4A16)**: the linear-projection weights (`q/k/v/o_proj`, `gate/up/down_proj`) are stored quantized with per-group FP16 scales (128 weights per scale) and dequantized to FP16 in-kernel; `int4` additionally packs two weights per byte. Token embeddings / LM head, RMSNorm weights, biases, Qwen3 QK-norm weights, and the KV cache remain FP16. This shrinks the projection-weight footprint to roughly 1/2 (`int8`) or 1/4 (`int4`), e.g. Qwen2.5-3B / Qwen3-4B fit comfortably in 8 GB VRAM where the FP16 model would not.

2. **Build the Engine**: Specify the model family, size, precision, and target GPU compute architecture (e.g., `75` for RTX 2070, `86` for RTX 3090, `89` for RTX 4090). All four are **required**.

```bash
mkdir -p build
cd build
cmake -Dmodel=<family> -Dmodelsize=<size> -Dquant=<quant> -Dgpu_arch=<gpu_architecture> ..
make -j$(nproc)
cd ..
```
where `family`/`size` **must match** what was exported, and `quant` is `fp16`, `int8` or `int4` and **must match the `--quantization` used during export** (the engine validates the family, dimensions, and precision against the binary header).

Example — Qwen3-0.6B in INT4 on an RTX 2070:

```bash
python3 export_weights.py --model=qwen3 --modelsize=0.6b --quantization=int4
mkdir -p build && cd build
cmake -Dmodel=qwen3 -Dmodelsize=0.6b -Dquant=int4 -Dgpu_arch=75 ..
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
python3 export_weights.py --model=<family> --modelsize=<size> --quantization=<quant>
```
2. **Build and Run Benchmark Binary** (all four arguments are **required**, and `family`/`size`/`quant` must match the exported weights):
```bash
cd benchmark
mkdir -p build
cd build
cmake -Dmodel=<family> -Dmodelsize=<size> -Dquant=<quant> -Dgpu_arch=<gpu_architecture> ..
make -j$(nproc)
cd ..
./build/cuqwen_benchmark
```
The `benchmark/` directory also has the scripts used to benchmark `vLLM` and `llama.cpp`. To run these benchmarking scripts please refer [here](https://github.com/talhatahir-10xe/CuQwen/blob/main/benchmark/README.md).

## Roadmap & Future Work
The following enhancements are planned for future development cycles:

* **Optimized Prefill Kernels:** Implement parallel prefill kernels to process prompt tokens in parallel rather than using the single-token autoregressive decode path for prefill stage.
* **Weight Quantization:** Add low-precision INT8 and INT4 (W8A16/W4A16) quantization. ✅ *Done in release 1.2 — weights-only INT8/INT4 for all model sizes; see the [Release 1.2 Benchmark](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.2_BENCHMARK.md).*
* **KV-Cache Quantization:** Support INT8/FP8 KV-cache quantization to cut attention memory bandwidth usage during long-context generation.
* **Modern Microarchitecture Tuning:** Extend custom kernel implementations for NVIDIA Hopper (`sm_90`) and Blackwell (`sm_100`) architectures utilizing `TMA` (Tensor Memory Accelerator), `DSMEM` (Distributed Shared memory) and `DPX` instructions.
* **Long-Context Throughput Optimization:** Further mitigate throughput decay across extended sequence lengths. ✅ *Done in release 1.1 — throughput decay across the 32K context window was cut roughly in half; see the [Release 1.1 Optimization](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_OPTIMIZATION.md) document.*
* **Support for Newer Qwen Architecture Series:** Add native kernel and engine support for newer iterations in the Qwen family. ✅ *Qwen3 dense (0.6B–14B) added in release 1.3 — per-head QK-RMSNorm, bias-free QKV, decoupled head_dim; Qwen 3.5 / 3.8 to follow.*

## Documentation

* [**Release 1.0 Benchmark**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.0_BENCHMARK.md): Benchmarking details and results of CuQwen 1.0 vs vLLM
* [**Release 1.0 Optimization Journey**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.0_OPTIMIZATION_JOURNEY.md): The complete journey of the different optimizations used in the process of implementing CuQwen 1.0
* [**Release 1.1 Benchmark**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_BENCHMARK.md): Benchmarking details and results of CuQwen 1.1 vs vLLM across the extended 32K context window
* [**Release 1.1 Optimization**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.1_OPTIMIZATION.md): The long-context optimizations applied in CuQwen 1.1 to reduce throughput decay over CuQwen 1.0
* [**Release 1.2 Benchmark**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.2_BENCHMARK.md): Benchmarking details and results of CuQwen 1.2 vs vLLM and llama.cpp across FP16/INT8/INT4 weights and the 32K context window
* [**Release 1.2 Optimization**](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.2_OPTIMIZATION.md): The weights-only INT8/INT4 quantization and decode-path kernel optimizations added in CuQwen 1.2 over CuQwen 1.1
