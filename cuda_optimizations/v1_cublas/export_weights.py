import os
import json
import struct
import numpy as np
from huggingface_hub import snapshot_download

# Constants & Configuration
MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
WEIGHTS_DIR     = "weights"
OUTPUT_BIN_PATH = os.path.join(WEIGHTS_DIR, "model_fp16_1_5b.bin")
OUTPUT_BPE_PATH = os.path.join(WEIGHTS_DIR, "qwen25_bpe_ranks.json")
MAGIC_NUMBER    = 0x5157454E

def print_header(title: str):
    print(f"\n╔═{'═' * 76}═╗")
    print(f"║ {title.center(76)} ║")
    print(f"╚═{'═' * 76}═╝")

def print_section(title: str):
    print(f"\n┌── {title} " + "─" * (74 - len(title)))


# HuggingFace & Safetensors I/O
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


# Exporters
def write_fp16(file_obj, arr: np.ndarray) -> int:
    data = arr.astype(np.float16).tobytes()
    file_obj.write(data)
    return len(data)


def export_weights(model_dir: str):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    cfg = read_config(model_dir)
    vocab_size          = cfg["vocab_size"]
    dim                 = cfg["hidden_size"]
    intermediate_size   = cfg["intermediate_size"]
    n_layers            = cfg["num_hidden_layers"]
    n_heads             = cfg["num_attention_heads"]
    n_kv_heads          = cfg["num_key_value_heads"]
    head_dim            = dim // n_heads
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
    print(f"  │ • Precision:        FP16")
    print(f"  │ • Output Path:      {OUTPUT_BIN_PATH}")

    reader = MultiShardSafetensors(model_dir)

    print_section("3. Serializing FP16 Binary Weights")
    total_bytes = 0

    with open(OUTPUT_BIN_PATH, "wb") as f:
        # Header (256 bytes)
        header = struct.pack(
            "iiiiiiiii",
            MAGIC_NUMBER, vocab_size, dim, intermediate_size,
            n_layers, n_heads, n_kv_heads, head_dim, max_seq_len,
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

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.q_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.q_proj.bias"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.k_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.k_proj.bias"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.v_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.v_proj.bias"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.self_attn.o_proj.weight"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.post_attention_layernorm.weight"))

            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.mlp.gate_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.mlp.up_proj.weight"))
            total_bytes += write_fp16(f, reader.get_tensor(f"{p}.mlp.down_proj.weight"))

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

    print(f"  │ [✔] Binary Export Complete → '{OUTPUT_BIN_PATH}' ({total_bytes / (1024**2):.2f} MB)")


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


# Main
if __name__ == "__main__":
    print_header("CUQWEN: FP16 WEIGHT & TOKENIZER EXPORTER")
    model_dir = download_model(MODEL_ID)
    export_weights(model_dir)
    export_tokenizer(model_dir)
    print_header("ALL EXPORTS COMPLETED SUCCESSFULLY")