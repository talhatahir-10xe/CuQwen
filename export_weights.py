import os
import sys
import json
import struct
import argparse
import numpy as np
from huggingface_hub import snapshot_download

# =============================================================================
# Constants & Model Mapping
# =============================================================================
# Model family -> { size -> HuggingFace repo }. Qwen3 dense repos are the base
# (hybrid-thinking) checkpoints; Qwen2.5 uses the Instruct checkpoints.
MODEL_MAP = {
    "qwen2.5": {
        "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
        "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
        "3b":   "Qwen/Qwen2.5-3B-Instruct",
        "7b":   "Qwen/Qwen2.5-7B-Instruct",
        "14b":  "Qwen/Qwen2.5-14B-Instruct",
    },
    "qwen3": {
        "0.6b": "Qwen/Qwen3-0.6B",
        "1.7b": "Qwen/Qwen3-1.7B",
        "4b":   "Qwen/Qwen3-4B",
        "8b":   "Qwen/Qwen3-8B",
        "14b":  "Qwen/Qwen3-14B",
    },
}

# Arch tag used in output filenames + arch id written to the binary header.
# Must match ARCH_TAG / arch_id in include/config.h.
ARCH_TAG = {"qwen2.5": "qwen25", "qwen3": "qwen3"}
ARCH_ID  = {"qwen2.5": 0, "qwen3": 1}

WEIGHTS_DIR     = "weights"
MAGIC_NUMBER    = 0x5157454E   # ASCII "QWEN"

# Quantization type tags written into the binary header (must match config.h)
QUANT_FP16      = 0
QUANT_INT8      = 1
QUANT_INT4      = 2
QUANT_GROUP     = 128          # weights per scale (grouped along the input/contraction dim)
QUANT_TYPE_ID   = {"fp16": QUANT_FP16, "int8": QUANT_INT8, "int4": QUANT_INT4}

def print_header(title: str):
    print(f"\n╔═{'═' * 76}═╗")
    print(f"║ {title.center(76)} ║")
    print(f"╚═{'═' * 76}═╝")

def print_section(title: str):
    print(f"\n┌── {title} " + "─" * (74 - len(title)))


# =============================================================================
# HuggingFace & Safetensors I/O
# =============================================================================
def download_model(repo_id: str) -> str:
    print_section("1. Fetching Model Artifacts from HuggingFace Hub")
    print(f"  │ Downloading '{repo_id}'...")
    local_dir = snapshot_download(
        repo_id=repo_id,
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  │ [✔] Model cached at: {local_dir}")
    return local_dir


def read_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        return json.load(f)


def load_safetensors(st_path: str):
    with open(st_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = json.loads(f.read(header_size))
        raw_bytes   = f.read()
    return header_json, raw_bytes


class MultiShardSafetensors:
    """Helper to handle single or multi-shard safetensors models seamlessly."""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.shards = {}
        
        st_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".safetensors")])
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_dir}")
        
        print(f"  │ Indexing {len(st_files)} safetensor shard(s)...")
        for st_file in st_files:
            st_path = os.path.join(model_dir, st_file)
            header_json, raw_bytes = load_safetensors(st_path)
            for tensor_name in header_json.keys():
                if tensor_name != "__metadata__":
                    self.shards[tensor_name] = (header_json, raw_bytes, tensor_name)

    def get_tensor(self, key: str) -> np.ndarray:
        if key not in self.shards:
            raise KeyError(f"Tensor '{key}' not found across safetensor shards.")
        
        header_json, raw_bytes, _ = self.shards[key]
        meta       = header_json[key]
        dtype      = meta["dtype"]
        shape      = meta["shape"]
        start, end = meta["data_offsets"]
        chunk      = raw_bytes[start:end]

        if dtype == "BF16":
            u16 = np.frombuffer(chunk, dtype=np.uint16)
            f32 = (u16.astype(np.uint32) << 16).view(np.float32)
            arr = f32.astype(np.float16)
        elif dtype == "F16":
            arr = np.frombuffer(chunk, dtype=np.float16).copy()
        elif dtype == "F32":
            arr = np.frombuffer(chunk, dtype=np.float32).astype(np.float16)
        else:
            raise ValueError(f"Unsupported safetensors dtype '{dtype}' for key '{key}'")

        return arr.reshape(shape)

    def contains(self, substring: str) -> bool:
        return any(substring in k for k in self.shards.keys())


# =============================================================================
# Exporters
# =============================================================================
def write_fp16(file_obj, arr: np.ndarray) -> int:
    data = arr.astype(np.float16).tobytes()
    file_obj.write(data)
    return len(data)


def quantize_int8_groupwise(arr: np.ndarray, group: int = QUANT_GROUP):
    """Symmetric per-group INT8 quantization along the input (contraction) dim.

    `arr` is a 2D weight matrix [out_features, in_features]. Each row is split
    into groups of `group` consecutive elements; every group gets one FP16 scale
    (= max(|w|) / 127). Returns (int8_weights [out, in], fp16_scales [out, in/group]).
    """
    arr = arr.astype(np.float32)
    out_features, in_features = arr.shape
    assert in_features % group == 0, (
        f"in_features={in_features} not divisible by group size {group}"
    )
    n_groups = in_features // group

    grouped = arr.reshape(out_features, n_groups, group)
    # Per-group max-abs; guard all-zero groups so we never divide by zero.
    max_abs = np.max(np.abs(grouped), axis=2)
    scales = max_abs / 127.0
    safe_scales = np.where(scales > 0.0, scales, 1.0)

    q = np.round(grouped / safe_scales[:, :, None])
    q = np.clip(q, -127, 127).astype(np.int8)

    return q.reshape(out_features, in_features), scales.astype(np.float16)


def write_int8(file_obj, arr: np.ndarray, group: int = QUANT_GROUP) -> int:
    """Serialize a weight matrix as [INT8 weights][FP16 group scales]."""
    if arr.ndim != 2:
        raise ValueError(f"INT8 export expects a 2D weight matrix, got shape {arr.shape}")
    q, scales = quantize_int8_groupwise(arr, group)
    w_bytes = q.tobytes()
    s_bytes = scales.astype(np.float16).tobytes()
    file_obj.write(w_bytes)
    file_obj.write(s_bytes)
    return len(w_bytes) + len(s_bytes)


def quantize_int4_groupwise(arr: np.ndarray, group: int = QUANT_GROUP):
    """Symmetric per-group INT4 quantization along the input (contraction) dim.

    Same scheme as INT8 but with a 4-bit range: scale = max(|w|) / 7, values
    clamped to [-7, 7]. Returns (int4_values [out, in] as int8 in [-7,7],
    fp16_scales [out, in/group]).
    """
    arr = arr.astype(np.float32)
    out_features, in_features = arr.shape
    assert in_features % group == 0, (
        f"in_features={in_features} not divisible by group size {group}"
    )
    n_groups = in_features // group

    grouped = arr.reshape(out_features, n_groups, group)
    max_abs = np.max(np.abs(grouped), axis=2)
    scales = max_abs / 7.0
    safe_scales = np.where(scales > 0.0, scales, 1.0)

    q = np.round(grouped / safe_scales[:, :, None])
    q = np.clip(q, -7, 7).astype(np.int8)

    return q.reshape(out_features, in_features), scales.astype(np.float16)


def write_int4(file_obj, arr: np.ndarray, group: int = QUANT_GROUP) -> int:
    """Serialize a weight matrix as [packed INT4 weights][FP16 group scales].

    Two weights are packed per byte: element 2p in the low nibble and element
    2p+1 in the high nibble (matching the in-kernel unpack in load_w8).
    """
    if arr.ndim != 2:
        raise ValueError(f"INT4 export expects a 2D weight matrix, got shape {arr.shape}")
    q, scales = quantize_int4_groupwise(arr, group)
    out_features, in_features = q.shape

    # 4-bit two's-complement nibbles (e.g. -7 -> 0x9), packed low|high per byte.
    nib = (q.astype(np.int16) & 0xF).astype(np.uint8).reshape(out_features, in_features // 2, 2)
    packed = (nib[:, :, 0] | (nib[:, :, 1] << 4)).astype(np.uint8)

    w_bytes = packed.tobytes()
    s_bytes = scales.astype(np.float16).tobytes()
    file_obj.write(w_bytes)
    file_obj.write(s_bytes)
    return len(w_bytes) + len(s_bytes)


def export_weights(model_dir: str, arch: str, model_size: str, quant: str):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    quant_tag    = quant
    quant_type   = QUANT_TYPE_ID[quant]
    arch_tag     = ARCH_TAG[arch]
    arch_id      = ARCH_ID[arch]
    is_quantized = (quant != "fp16")
    out_bin_path = os.path.join(
        WEIGHTS_DIR, f"model_{arch_tag}_{quant_tag}_{model_size.replace('.', '_')}.bin"
    )

    # For quantized builds, only the linear-projection weights are quantized;
    # everything else (embeddings/LM head, RMSNorm weights, biases) stays FP16.
    def write_proj(f, arr):
        if quant == "int8":
            return write_int8(f, arr)
        if quant == "int4":
            return write_int4(f, arr)
        return write_fp16(f, arr)

    cfg = read_config(model_dir)
    vocab_size          = cfg["vocab_size"]
    dim                 = cfg["hidden_size"]
    intermediate_size   = cfg["intermediate_size"]
    n_layers            = cfg["num_hidden_layers"]
    n_heads             = cfg["num_attention_heads"]
    n_kv_heads          = cfg["num_key_value_heads"]
    head_dim            = cfg.get("head_dim", dim // n_heads)
    max_seq_len         = cfg.get("max_position_embeddings", 32768)
    tie_word_embeddings = int(cfg.get("tie_word_embeddings", True))

    print_section("2. Model Specifications & Output Target")
    print(f"  │ • Vocab Size:       {vocab_size:,}")
    print(f"  │ • Hidden Dim:       {dim}")
    print(f"  │ • Intermediate Dim: {intermediate_size}")
    print(f"  │ • Layers:           {n_layers}")
    print(f"  │ • Query Heads:      {n_heads}")
    print(f"  │ • Key/Value Heads:  {n_kv_heads}")
    print(f"  │ • Head Dim:         {head_dim}")
    print(f"  │ • Max Seq Len:      {max_seq_len:,}")
    print(f"  │ • Tied Embeddings:  {bool(tie_word_embeddings)}")
    if quant == "int8":
        print(f"  │ • Precision:        INT8 weights-only (W8A16), group={QUANT_GROUP}")
        print(f"  │ • FP16 retained:    embeddings/LM head, RMSNorm weights, biases")
    elif quant == "int4":
        print(f"  │ • Precision:        INT4 weights-only (W4A16, packed), group={QUANT_GROUP}")
        print(f"  │ • FP16 retained:    embeddings/LM head, RMSNorm weights, biases")
    else:
        print(f"  │ • Precision:        FP16")
    print(f"  │ • Output Path:      {out_bin_path}")

    reader = MultiShardSafetensors(model_dir)

    # Derive the attention layout from the checkpoint itself (robust across both
    # families): Qwen2.5 carries q/k/v_proj biases and no QK-norm; Qwen3 drops the
    # biases and adds per-head q_norm/k_norm. The C++ loader uses the same order.
    has_qkv_bias = reader.contains("self_attn.q_proj.bias")
    has_qk_norm  = reader.contains("self_attn.q_norm")
    print(f"  │ • QKV bias:         {has_qkv_bias}")
    print(f"  │ • QK-Norm:          {has_qk_norm}")

    # The LM head is the single largest tensor read every decode step. When it is
    # a SEPARATE (untied) tensor and we are building a quantized model, store it
    # INT8 (W8A16) too: this halves its per-token bandwidth with negligible logit
    # error. Tied models reuse the (FP16) embedding table, so they are unaffected.
    lm_head_is_int8 = is_quantized and reader.contains("lm_head") and not tie_word_embeddings
    lm_head_fmt = QUANT_INT8 if lm_head_is_int8 else QUANT_FP16

    print_section(f"3. Serializing {quant.upper()} Binary Weights")
    total_bytes = 0

    with open(out_bin_path, "wb") as f:
        # Header (256 bytes). Field 10 (lm_head_fmt) flags the LM-head precision
        # (0 = FP16/tied, 1 = INT8); field 11 (arch_id) is the family (0 = Qwen2.5,
        # 1 = Qwen3). Older files have 0 in fields 10/11 from zero padding.
        header = struct.pack(
            "iiiiiiiiiiii",
            MAGIC_NUMBER, vocab_size, dim, intermediate_size,
            n_layers, n_heads, n_kv_heads, head_dim, max_seq_len,
            quant_type, lm_head_fmt, arch_id,
        )

        f.write(header)
        f.write(b"\x00" * (256 - len(header)))
        total_bytes += 256

        # Token embeddings (FP16)
        print("  │ [+] Exporting embed_tokens.weight...")
        total_bytes += write_fp16(f, reader.get_tensor("model.embed_tokens.weight"))

        # Transformer layers
        for l in range(n_layers):
            p = f"model.layers.{l}"
            print(f"  │ [+] Exporting Layer {l:2d}/{n_layers - 1} ...", end="\r")

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.input_layernorm.weight"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.q_proj.weight"))
            if has_qkv_bias:
                total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.q_proj.bias"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.k_proj.weight"))
            if has_qkv_bias:
                total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.k_proj.bias"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.v_proj.weight"))
            if has_qkv_bias:
                total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.v_proj.bias"))

            # Qwen3: per-head QK-RMSNorm weights (FP16), placed after the QKV
            # projections and before o_proj — matching the C++ loader order.
            if has_qk_norm:
                total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.q_norm.weight"))
                total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.k_norm.weight"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.o_proj.weight"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.post_attention_layernorm.weight"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.mlp.gate_proj.weight"))
            total_bytes += write_proj(f, reader.get_tensor(f"{p}.mlp.up_proj.weight"))
            total_bytes += write_proj(f, reader.get_tensor(f"{p}.mlp.down_proj.weight"))

        print()

        # Final RMSNorm (FP16)
        print("  │ [+] Exporting final model.norm.weight...")
        total_bytes += write_fp16(f, reader.get_tensor("model.norm.weight"))

        # LM Head: INT8 for quantized untied models, else FP16 (or tied/skipped).
        if lm_head_is_int8:
            print("  │ [+] Exporting lm_head.weight (INT8, W8A16)...")
            total_bytes += write_int8(f, reader.get_tensor("lm_head.weight"))
        elif reader.contains("lm_head") and not tie_word_embeddings:
            print("  │ [+] Exporting lm_head.weight (FP16)...")
            total_bytes += write_fp16(f, reader.get_tensor("lm_head.weight"))
        else:
            print("  │ [+] lm_head tied to embed_tokens — skipping duplicate export")

    print(f"  │ [✔] Binary Export Complete → '{out_bin_path}' ({total_bytes / (1024**2):.2f} MB)")


def export_tokenizer(model_dir: str, out_bpe_path: str):
    print_section("4. Exporting Tokenizer Resources")
    vocab_path     = os.path.join(model_dir, "vocab.json")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    tc_path        = os.path.join(model_dir, "tokenizer_config.json")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    added_tokens = {}
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        for entry in tok_data.get("added_tokens", []):
            added_tokens[int(entry["id"])] = entry["content"]

    special_map = {}
    if os.path.exists(tc_path):
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        for k, v in tc.items():
            if isinstance(v, str) and v in vocab:
                special_map[k] = v
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item in vocab:
                        special_map[item] = item

    data = {
        "vocab":          vocab,
        "special_tokens": special_map,
        "added_tokens":   added_tokens,
    }

    with open(out_bpe_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  │ [✔] Tokenizer JSON Export Complete → '{out_bpe_path}'")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen2.5 / Qwen3 PyTorch/Safetensors weights and tokenizer to CuQwen binary format.")
    parser.add_argument("--model", type=str, required=True, choices=["qwen2.5", "qwen3"],
                        help="Model family / architecture (qwen2.5 or qwen3)")
    parser.add_argument("--modelsize", type=str, required=True,
                        help="Model size within the family (qwen2.5: 0.5b,1.5b,3b,7b,14b | qwen3: 0.6b,1.7b,4b,8b,14b)")
    parser.add_argument("--quantization", type=str, required=True, choices=["fp16", "int8", "int4"],
                        help="Weight precision: 'fp16' (W16A16), 'int8' (W8A16) or 'int4' (W4A16), weights-only")

    if len(sys.argv) == 1:
        parser.print_help()
        print("\n--model, --modelsize and --quantization are required. Example commands:")
        print("  python3 export_weights.py --model=qwen3   --modelsize=0.6b --quantization=int4")
        print("  python3 export_weights.py --model=qwen3   --modelsize=1.7b --quantization=int8")
        print("  python3 export_weights.py --model=qwen2.5 --modelsize=3b   --quantization=fp16")
        sys.exit(1)

    args = parser.parse_args()

    if args.modelsize not in MODEL_MAP[args.model]:
        valid = ", ".join(MODEL_MAP[args.model].keys())
        print(f"[!] Invalid --modelsize '{args.modelsize}' for --model '{args.model}'. Valid sizes: {valid}")
        sys.exit(1)

    print_header(f"CUQWEN: {args.model.upper()} {args.quantization.upper()} WEIGHT & TOKENIZER EXPORTER")
    repo_id = MODEL_MAP[args.model][args.modelsize]
    out_bpe_path = os.path.join(WEIGHTS_DIR, f"{ARCH_TAG[args.model]}_bpe_ranks.json")
    model_dir = download_model(repo_id)
    export_weights(model_dir, args.model, args.modelsize, args.quantization)
    export_tokenizer(model_dir, out_bpe_path)
    print_header("ALL EXPORTS COMPLETED SUCCESSFULLY")
