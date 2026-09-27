# CuQwen Benchmarking — Commands & Scheme Reference

Everything needed to run the **application** (coherence check) and the **sampled
decode benchmark** for **vLLM** and **Ollama**, across every model size
(`0.5b`, `1.5b`, `3b`, `7b`) and precision (`fp16`, `int8`, `int4`). A short
CuQwen section is included at the end for the full 3-way comparison.

All engines are benchmarked identically: **9 sampled context depths** (a 0.5K
decode slice ending at ~1/5/9/13/17/21/25/29/32K), timing the **decode phase
only** (prefill excluded).

---

## Quantization scheme — what each engine actually does

| Aspect | **CuQwen** int8/int4 | **vLLM** GPTQ Int8/Int4 | **Ollama** Q8_0 / Q4_0 |
|---|---|---|---|
| Source | built by `export_weights.py` (RTN) | official `Qwen/…-GPTQ-Int8/Int4` | official `…-GGUF` q8_0/q4_0 |
| Quant algorithm | RTN, symmetric | GPTQ, symmetric (`sym=True`) | RTN, symmetric |
| Group / block size | **128** | **128** | **32** |
| Scale dtype | FP16 | FP16 | FP16 |
| Zero point | none | none | none |
| **Matmul compute** | **W8A16 / W4A16** (FP16 activations; weights → FP16) | **W8A16 / W4A16** (FP16 activations) | **W8A8 / W4A8** (activations quantized to INT8 Q8_1; integer `dp4a` dot) |
| KV cache | FP16 | FP16 | FP16 |
| Token embeddings | FP16 | FP16 | quantized (Q8_0 / Q4_0) |
| lm_head (output) | FP16 | FP16 | quantized (Q8_0 / **Q6_K**) |
| Norms (RMSNorm) | FP16 | FP16 | F32 |

**Takeaways for a fair comparison**
- **fp16**: all three are identical W16A16 — a clean 3-way baseline.
- **int8/int4**: **CuQwen ↔ vLLM are apples-to-apples** (both true W*A16, group-128,
  symmetric, no zero-point, FP16 embeddings/lm_head/norms). They differ only in the
  *calibration algorithm* (RTN vs GPTQ), which affects quality, not the storage
  format or the kernels (hence not decode speed).
- **Ollama is a different scheme** and should be reported as such: block size 32,
  quantized embeddings/lm_head, F32 norms, and — most importantly — it is
  **W8A8 / W4A8** (it dynamically quantizes activations to INT8 and does integer
  matmuls), *not* weights-only W*A16. GGUF offers no group-128 symmetric W*A16 type.

---

## Environment (shared venv for vLLM + Ollama scripts)

```bash
cd ~/Downloads/CuQwen                       # repo root
python3 -m venv ~/cuqwen-vllm-venv
~/cuqwen-vllm-venv/bin/pip install --upgrade pip
~/cuqwen-vllm-venv/bin/pip install "vllm==0.26.0" ollama huggingface_hub
```
(`vllm` pulls a compatible torch + transformers; `ollama` is the Python client; all
Qwen2.5 models are public, no HF token needed.)

---

# 1) vLLM

vLLM **auto-downloads** the official Qwen2.5 models on first run — no build/convert
step. `--quantization` maps to: `fp16` → base, `int8` → `GPTQ-Int8`, `int4` → `GPTQ-Int4`.

### Application (coherence)
```bash
cd ~/Downloads/CuQwen/benchmark
~/cuqwen-vllm-venv/bin/python vllm_application.py --model=1.5b --quantization=int8
# type prompts at "User >", then "exit"
```

### Benchmark
```bash
cd ~/Downloads/CuQwen/benchmark
for size in 0.5b 1.5b 3b 7b; do
  for q in fp16 int8 int4; do
    echo "############## vLLM  $size / $q ##############"
    ~/cuqwen-vllm-venv/bin/python vllm_benchmark.py --model=$size --quantization=$q
  done
done
```
> On Ampere+ (RTX 3090 = sm_86) vLLM uses Marlin kernels for GPTQ. On older Turing
> (sm_75) it falls back to a slower GPTQ path; fp16 at 32K may not fit 8 GB but is
> fine on the 3090's 24 GB.

---

# 2) Ollama

Model tags the scripts expect: `qwen2.5:<size>-fp16`, `qwen2.5:<size>-q8_0`,
`qwen2.5:<size>-q4_0` (`--quantization=int8 → q8_0`, `int4 → q4_0`).

### One-time engine setup
```bash
apt-get update && apt-get install -y curl zstd
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_NUM_PARALLEL=1
ollama serve > /dev/null 2>&1 &
mkdir -p ~/qwen_gguf && cd ~/qwen_gguf
```
(`hf` comes with `huggingface_hub`; if not on PATH use `~/cuqwen-vllm-venv/bin/hf`.)

### Download GGUFs + create models

**0.5B** (all single-file)
```bash
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-fp16.gguf --local-dir .
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir .
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_0.gguf --local-dir .
echo "FROM ./qwen2.5-0.5b-instruct-fp16.gguf" > Modelfile && ollama create qwen2.5:0.5b-fp16 -f Modelfile
echo "FROM ./qwen2.5-0.5b-instruct-q8_0.gguf" > Modelfile && ollama create qwen2.5:0.5b-q8_0 -f Modelfile
echo "FROM ./qwen2.5-0.5b-instruct-q4_0.gguf" > Modelfile && ollama create qwen2.5:0.5b-q4_0 -f Modelfile
```

**1.5B** (all single-file)
```bash
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf --local-dir .
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf --local-dir .
hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_0.gguf --local-dir .
echo "FROM ./qwen2.5-1.5b-instruct-fp16.gguf" > Modelfile && ollama create qwen2.5:1.5b-fp16 -f Modelfile
echo "FROM ./qwen2.5-1.5b-instruct-q8_0.gguf" > Modelfile && ollama create qwen2.5:1.5b-q8_0 -f Modelfile
echo "FROM ./qwen2.5-1.5b-instruct-q4_0.gguf" > Modelfile && ollama create qwen2.5:1.5b-q4_0 -f Modelfile
```

**3B** (fp16 = 2 shards; q8_0 / q4_0 single-file)
```bash
hf download Qwen/Qwen2.5-3B-Instruct-GGUF --include "qwen2.5-3b-instruct-fp16-*" --local-dir .
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q8_0.gguf --local-dir .
hf download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_0.gguf --local-dir .
cat > Modelfile <<'EOF'
FROM ./qwen2.5-3b-instruct-fp16-00001-of-00002.gguf
FROM ./qwen2.5-3b-instruct-fp16-00002-of-00002.gguf
EOF
ollama create qwen2.5:3b-fp16 -f Modelfile
echo "FROM ./qwen2.5-3b-instruct-q8_0.gguf" > Modelfile && ollama create qwen2.5:3b-q8_0 -f Modelfile
echo "FROM ./qwen2.5-3b-instruct-q4_0.gguf" > Modelfile && ollama create qwen2.5:3b-q4_0 -f Modelfile
```

**7B** (all sharded: fp16 ×4, q8_0 ×3, q4_0 ×2)
```bash
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-fp16-*" --local-dir .
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q8_0-*" --local-dir .
hf download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_0-*" --local-dir .
cat > Modelfile <<'EOF'
FROM ./qwen2.5-7b-instruct-fp16-00001-of-00004.gguf
FROM ./qwen2.5-7b-instruct-fp16-00002-of-00004.gguf
FROM ./qwen2.5-7b-instruct-fp16-00003-of-00004.gguf
FROM ./qwen2.5-7b-instruct-fp16-00004-of-00004.gguf
EOF
ollama create qwen2.5:7b-fp16 -f Modelfile
cat > Modelfile <<'EOF'
FROM ./qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
FROM ./qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf
FROM ./qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf
EOF
ollama create qwen2.5:7b-q8_0 -f Modelfile
cat > Modelfile <<'EOF'
FROM ./qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf
FROM ./qwen2.5-7b-instruct-q4_0-00002-of-00002.gguf
EOF
ollama create qwen2.5:7b-q4_0 -f Modelfile
```
Verify: `ollama list`

### Application (coherence)
```bash
cd ~/Downloads/CuQwen/benchmark
~/cuqwen-vllm-venv/bin/python ollama_application.py --model=1.5b --quantization=int8
```

### Benchmark
```bash
cd ~/Downloads/CuQwen/benchmark
for size in 0.5b 1.5b 3b 7b; do
  for q in fp16 int8 int4; do
    echo "############## Ollama  $size / $q ##############"
    ~/cuqwen-vllm-venv/bin/python ollama_benchmark.py --model=$size --quantization=$q
  done
done
```
> Ollama reports tok/s as the server's pure per-token decode rate
> (`eval_count / eval_duration`), stable regardless of how many tokens a window
> decoded (Ollama has no `ignore_eos`, so decode length varies). On 8 GB GPUs fp16
> may cap `num_ctx` (deep windows warn `[!] depth << target`); a 24 GB 3090 reaches 32K.

---

# 3) CuQwen (for the 3-way comparison)

```bash
cd ~/Downloads/CuQwen
# Export weights (per size/precision)
python3 export_weights.py --model=1.5b --quantization=int8    # fp16 | int8 | int4

# Build (gpu_arch: 75=RTX2070, 86=RTX3090, 89=RTX4090; -Dquant must match export)
cd benchmark && mkdir -p build && cd build
cmake -Dmodel=1.5b -Dgpu_arch=86 -Dquant=int8 .. && make -j$(nproc) && cd ..
./build/cuqwen_benchmark
```
