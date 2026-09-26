"""
Produce weights-only INT8 (W8A16) / INT4 (W4A16) compressed-tensors checkpoints
for vLLM that match CuQwen's quantization scheme, so the two engines can be
benchmarked head-to-head fairly.

Scheme (identical to CuQwen's export_weights.py):
  • Symmetric, per-group quantization with group_size = 128 along the input
    (contraction) dim, NO zero point.
  • scale = max(|w|) / qmax  (qmax = 127 for int8, 7 for int4), stored FP16.
  • Only Linear projection weights are quantized; `lm_head` is ignored and
    embeddings / RMSNorm weights / biases stay FP16 (they are not Linear).
  • Activations and the KV cache remain FP16 (weights-only).

Output: weights/vllm_quant/<model>_<quant>/  (a standard HF/compressed-tensors
model directory that vLLM loads directly).

Usage:
  python3 quantize_for_vllm.py --model=1.5b --quantization=int8
  python3 quantize_for_vllm.py --model=1.5b --quantization=int4
"""
import os
import argparse
import torch

# compressed-tensors >=0.18 references torch.nn.Buffer (added in torch 2.5).
# Shim it for older torch (2.4) so compress_model()'s state-dict walk works.
if not hasattr(torch.nn, "Buffer"):
    torch.nn.Buffer = torch.Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer
from compressed_tensors.quantization import (
    apply_quantization_config, QuantizationConfig, QuantizationStatus, preset_name_to_scheme,
)
from compressed_tensors import ModelCompressor, CompressionFormat

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}

GROUP_SIZE = 128


def quant_output_dir(model_size: str, quant: str) -> str:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights", "vllm_quant")
    return os.path.normpath(os.path.join(root, f"{model_size.replace('.', '_')}_{quant}"))


def group_scales(weight: torch.Tensor, num_bits: int, group: int = GROUP_SIZE) -> torch.Tensor:
    """RTN symmetric per-group scale = max(|w|)/qmax, computed exactly like CuQwen."""
    qmax = (1 << (num_bits - 1)) - 1              # 127 (int8) or 7 (int4)
    out_features, in_features = weight.shape
    assert in_features % group == 0, f"in_features={in_features} not divisible by {group}"
    w = weight.detach().float().reshape(out_features, in_features // group, group)
    max_abs = w.abs().amax(dim=2)
    scale = (max_abs / qmax).clamp(min=1e-8)      # guard all-zero groups
    return scale.to(torch.float16)


def main():
    parser = argparse.ArgumentParser(description="Produce W8A16/W4A16 compressed-tensors model for vLLM")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--quantization", required=True, choices=["int8", "int4"])
    args = parser.parse_args()

    model_id = MODEL_MAP[args.model]
    num_bits = 8 if args.quantization == "int8" else 4
    scheme_name = "W8A16" if args.quantization == "int8" else "W4A16"
    # compressed-tensors packs group-quantized int weights (4- and 8-bit) into
    # int32 via the pack-quantized format; vLLM's WNA16 path loads both.
    fmt = CompressionFormat.pack_quantized.value
    out_dir = quant_output_dir(args.model, args.quantization)

    print(f"[*] Model      : {model_id}")
    print(f"[*] Scheme     : {scheme_name} (weights-only, symmetric, group={GROUP_SIZE}, no zero-point)")
    print(f"[*] Format     : {fmt}")
    print(f"[*] Output dir : {out_dir}")

    print("[*] Loading FP16 model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.eval()

    # Build the compressed-tensors quantization config. The preset W8A16 / W4A16
    # schemes are already symmetric, group_size=128 in compressed-tensors >=0.18.
    scheme = preset_name_to_scheme(scheme_name, targets=["Linear"])
    config = QuantizationConfig(
        config_groups={"group_0": scheme},
        ignore=["lm_head"],
        quant_method="compressed-tensors",
        format=fmt,
        quantization_status=QuantizationStatus.FROZEN,
    )

    print("[*] Applying quantization config (initializes scale params)...")
    apply_quantization_config(model, config)

    print("[*] Computing RTN group scales from weights...")
    n_quantized = 0
    for _, module in model.named_modules():
        if hasattr(module, "weight_scale") and hasattr(module, "weight") and module.weight.dim() == 2:
            scale = group_scales(module.weight.data, num_bits)
            module.weight_scale.data = scale.to(
                dtype=module.weight_scale.dtype, device=module.weight_scale.device
            )
            if hasattr(module, "weight_zero_point") and module.weight_zero_point is not None:
                module.weight_zero_point.data.zero_()   # symmetric -> no zero point
            n_quantized += 1
    print(f"    [+] Set scales for {n_quantized} Linear layers")

    print("[*] Compressing (quantize + pack) and saving...")
    compressor = ModelCompressor.from_pretrained_model(model, quantization_format=fmt)
    compressor.compress_model(model)

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    compressor.update_config(out_dir)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(out_dir)

    print(f"[✔] Done. Quantized model written to: {out_dir}")


if __name__ == "__main__":
    main()
