# CuQwen: CUDA C++ Optimization Journey
This directory traces the iterative evolution of building a high-throughput, bare-metal C++/CUDA inference engine for **Qwen2.5-1.5B-Instruct**. Beginning with baseline cuBLAS execution, each stage systematically addresses hardware bottlenecks revealed through NVIDIA Nsight Systems and Nsight Compute profiling.

## Progression Matrix

| Optimization Phase | Primary Bottleneck Solved | Core CUDA Technique Applied |
| :--- | :---: | ---: |
| v1 (cublas) | Execution Baseline | Direct `cublasHgemm` calls |
| v2 (Kernel Fusion) | Memory bandwidth & Launch Overhead | Fused custom cuda kernels |
| v3 (Vectorized Memory) | Global Memory Alignment | 128-bit (`half2`/ `float4`) Vectorized Access |
| v4 (Warp Optimization) | Block Synchronization Contention | Register Shuffle Primitives (`__shfl`) |
| v5 (FlashDecode) | Long-Context KV Attention Latency | Split-K KV-Cache Partitioning |
| v6 (Row Tiling) | Suboptimal Arithmetic Intensity | Tile Multi-Row Processing & `__ldg` Caching |
| v7 (CUDA Graphs) | Host-side Driver Launch Latency | Static Graph Capture (`cudaGraphExec_t`) |

For build instructions and benchmark scripts, refer to the individual `README.md` files located inside each version directory.