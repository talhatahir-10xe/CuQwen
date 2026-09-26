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
MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct"
}

WEIGHTS_DIR     = "weights"
OUTPUT_BPE_PATH = os.path.join(WEIGHTS_DIR, "qwen25_bpe_ranks.json")
MAGIC_NUMBER    = 0x5157454E   # ASCII "QWEN"

# Quantization type tags written into the binary header (must match config.h)
QUANT_FP16      = 0
QUANT_INT8      = 1
QUANT_GROUP     = 128          # weights per scale (grouped along the input/contraction dim)

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


def export_weights(model_dir: str, model_size: str, quant: str = "fp16"):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    quant_tag  = "int8" if quant == "int8" else "fp16"
    quant_type = QUANT_INT8 if quant == "int8" else QUANT_FP16
    is_int8    = (quant == "int8")
    out_bin_path = os.path.join(WEIGHTS_DIR, f"model_{quant_tag}_{model_size.replace('.', '_')}.bin")

    # For INT8 builds, the linear-projection weights are quantized; everything
    # else (embeddings/LM head, RMSNorm weights, biases) stays FP16.
    def write_proj(f, arr):
        return write_int8(f, arr) if is_int8 else write_fp16(f, arr)

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
    if is_int8:
        print(f"  │ • Precision:        INT8 weights-only (W8A16), group={QUANT_GROUP}")
        print(f"  │ • FP16 retained:    embeddings/LM head, RMSNorm weights, biases")
    else:
        print(f"  │ • Precision:        FP16")
    print(f"  │ • Output Path:      {out_bin_path}")

    reader = MultiShardSafetensors(model_dir)

    print_section(f"3. Serializing {'INT8' if is_int8 else 'FP16'} Binary Weights")
    total_bytes = 0

    with open(out_bin_path, "wb") as f:
        # Header (256 bytes)
        header = struct.pack(
            "iiiiiiiiii",
            MAGIC_NUMBER, vocab_size, dim, intermediate_size,
            n_layers, n_heads, n_kv_heads, head_dim, max_seq_len,
            quant_type,
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
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.q_proj.bias"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.k_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.k_proj.bias"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.v_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.v_proj.bias"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.self_attn.o_proj.weight"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.post_attention_layernorm.weight"))

            total_bytes += write_proj(f, reader.get_tensor(f"{p}.mlp.gate_proj.weight"))
            total_bytes += write_proj(f, reader.get_tensor(f"{p}.mlp.up_proj.weight"))
            total_bytes += write_proj(f, reader.get_tensor(f"{p}.mlp.down_proj.weight"))

        print()

        # Final RMSNorm (FP16)
        print("  │ [+] Exporting final model.norm.weight...")
        total_bytes += write_fp16(f, reader.get_tensor("model.norm.weight"))

        # LM Head (FP16)
        if reader.contains("lm_head") and not tie_word_embeddings:
            print("  │ [+] Exporting lm_head.weight...")
            total_bytes += write_fp16(f, reader.get_tensor("lm_head.weight"))
        else:
            print("  │ [+] lm_head tied to embed_tokens — skipping duplicate export")

    print(f"  │ [✔] Binary Export Complete → '{out_bin_path}' ({total_bytes / (1024**2):.2f} MB)")


def export_tokenizer(model_dir: str):
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

    with open(OUTPUT_BPE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  │ [✔] Tokenizer JSON Export Complete → '{OUTPUT_BPE_PATH}'")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen 2.5 PyTorch/Safetensors weights and tokenizer to binary format.")
    parser.add_argument("--model", type=str, required=True, choices=["0.5b", "1.5b", "3b", "7b"],
                        help="Model size variant to export (0.5b, 1.5b, 3b, 7b)")
    parser.add_argument("--quantization", type=str, default="fp16", choices=["fp16", "int8"],
                        help="Weight precision: 'fp16' (default) or 'int8' (weights-only W8A16)")

    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExample commands:")
        print("  python3 export_weights.py --model=0.5b")
        print("  python3 export_weights.py --model=1.5b --quantization=int8")
        print("  python3 export_weights.py --model=3b  --quantization=int8")
        print("  python3 export_weights.py --model=7b")
        sys.exit(1)

    args = parser.parse_args()

    print_header(f"CUQWEN: {args.quantization.upper()} WEIGHT & TOKENIZER EXPORTER")
    repo_id = MODEL_MAP[args.model]
    model_dir = download_model(repo_id)
    export_weights(model_dir, args.model, args.quantization)
    export_tokenizer(model_dir)
    print_header("ALL EXPORTS COMPLETED SUCCESSFULLY")
