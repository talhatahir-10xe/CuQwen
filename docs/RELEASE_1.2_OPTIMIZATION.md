# CuQwen 1.2 Optimization: Weights-Only Quantization & Decode-Path Tuning

Where CuQwen 1.1 was a single-goal release (flatten long-context throughput decay), **CuQwen 1.2** has two threads of work: it adds **weights-only INT8/INT4 quantization**, and it tunes the decode path so the quantized kernels — and the FP16 path — run faster and port cleanly across GPUs. All of the 1.1 long-context work (fixed-partition FlashDecoding, Tensor-core attention, GQA-aware head grouping) is carried forward unchanged.

For the head-to-head throughput, speedup, and decay results these changes produce, see the [CuQwen 1.2 Benchmark](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.2_BENCHMARK.md).

---

## 1. Weights-Only Quantization (W8A16 / W4A16)

The linear-projection weights (`q/k/v/o_proj`, `gate/up/down_proj`) can now be stored and streamed as **INT8 (W8A16)** or **INT4 (W4A16)**, selected at export (`--quantization`) and build (`-Dquant`) time. Everything else — token embeddings, RMSNorm weights, biases, and the **KV cache** — stays FP16.

* **Symmetric per-group quantization** with a group size of 128 along the contraction dimension: each group of 128 weights shares one FP16 scale (`max|w|/127` for INT8, `max|w|/7` for INT4). This keeps quantization error low while adding negligible scale overhead.
* **Packed INT4 storage.** Two 4-bit weights share a byte (low/high nibble), so 8 INT4 weights = 4 bytes = a single 32-bit coalesced load, and are sign-extended and dequantized in-kernel.
* **Smaller footprint.** The projection-weight bytes shrink to roughly **1/2** (INT8) or **1/4** (INT4) — e.g. Qwen2.5-3B fits comfortably in 8 GB VRAM where the FP16 model does not. Because decode is memory-bandwidth bound, less weight traffic per token also means higher throughput.

---

## 2. Making the Quantized Path Fast: the `dot8` Seam

Quantization only helps if the in-kernel dequantization is cheap. The weight fetch + dequant + dot-product for every GEMV-style kernel is funneled through **one helper, `dot8`**, which is the *only* place FP16/INT8/INT4 handling differs — every kernel's reduction loop is identical across builds. Two changes there matter most:

* **FP32 accumulation, no FP16 round-trip.** 1.1 dequantized each weight into FP16 (`load_w8` → `half2`) before the FMA. 1.2 dequantizes `int → float` and accumulates the whole 8-wide dot product directly in FP32, removing the intermediate half conversions.
* **Scale applied once.** The shared per-group FP16 scale is multiplied into the 8-wide partial sum a single time, instead of scaling every individual weight.

Both cut the per-element ALU work that bottlenecks the low-bit kernels: INT4/INT8 read so few weight bytes that they are **dequant-bound, not memory-bound**, so trimming the dequant math is what turns the smaller footprint into real speed. `dot8` covers attention, both MLP stages, the `o_proj`/`down_proj` GEMVs, and the logits projection.

---

## 3. Other Decode-Path Optimizations

* **INT8 LM head for untied models.** On models whose LM head is a separate tensor (7B), it is stored INT8 and read through a dedicated `compute_logits_int8` kernel. The LM head is the single largest tensor streamed every decode step, so halving its traffic directly cuts per-token latency; tied models reuse the FP16 embedding table and are unaffected.
* **Wider blocks for the wide GEMVs.** `o_proj` and `down_proj` have few output rows but a large contraction dimension. On wide models (`dim ≥ 3072`, i.e. 7B) they now pack 16 warps per block instead of 8, improving L2 reuse of the shared input vector for roughly a **6%** lift on those kernels; narrower models keep 8 warps to avoid underfilling the GPU.
* **Parallel two-stage argmax.** Greedy sampling over the vocabulary moved from a single-block reduction to a **two-stage, SM-scaled** reduction that packs each `(logit, index)` pair into one 64-bit key (order-preserving float map, ties broken toward the lowest index). This keeps the final sampling step from becoming a serialization point on large-vocab models.
* **Centralized, per-GPU tuning knobs.** The performance-critical launch parameters (`ATTN_PARTITIONS`, `ARGMAX_BLOCKS`, the wide-GEMV threshold, warps-per-block) are now collected and documented in `config.h` with recommended values that scale with SM count (RTX 2070 → 3090 → 4090 → 5090), so the engine can be retuned for a new GPU without touching kernel code.

---

These changes make quantized CuQwen fast rather than merely smaller, while giving the FP16 path a modest lift on the largest model and a cleaner sampling tail. The resulting throughput across FP16/INT8/INT4 and the full 32K window is documented in the [CuQwen 1.2 Benchmark](https://github.com/talhatahir-10xe/CuQwen/blob/main/docs/RELEASE_1.2_BENCHMARK.md).
