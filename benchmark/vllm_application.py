"""
Simple vLLM chat application — the vLLM counterpart to CuQwen's interactive
chat (src/main.cu / ./build/cuqwen). Its purpose is to sanity-check that the
FP16 / INT8 (W8A16) / INT4 (W4A16) models produce coherent output under vLLM,
using the same quantized checkpoints that the benchmark uses.

Usage:
  python3 vllm_application.py --model=1.5b --quantization=fp16
  python3 vllm_application.py --model=1.5b --quantization=int8
  python3 vllm_application.py --model=1.5b --quantization=int4

Type a message at the "User >" prompt; type 'exit' or 'quit' (or send EOF) to
leave. Multi-turn conversation history is preserved within a session.
"""
import os
import sys
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from vllm import LLM, SamplingParams

MODEL_MAP = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
    "7b":   "Qwen/Qwen2.5-7B-Instruct",
}


def resolve_model(model_size: str, quant: str) -> str:
    """FP16 -> HF model id; int8/int4 -> local compressed-tensors dir."""
    if quant == "fp16":
        return MODEL_MAP[model_size]
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(
        here, "..", "weights", "vllm_quant", f"{model_size.replace('.', '_')}_{quant}"
    ))
    if not os.path.isdir(path):
        sys.exit(
            f"[!] Quantized model not found: {path}\n"
            f"    Produce it first:\n"
            f"      python3 quantize_for_vllm.py --model={model_size} --quantization={quant}"
        )
    return path


def main():
    parser = argparse.ArgumentParser(description="vLLM interactive chat (FP16/INT8/INT4)")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()),
                        help="Model size variant: 0.5b, 1.5b, 3b, or 7b")
    parser.add_argument("--quantization", default="fp16", choices=["fp16", "int8", "int4"],
                        help="Weight precision: fp16 (default), int8 (W8A16) or int4 (W4A16)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per reply")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Context length")
    args = parser.parse_args()

    model_id = resolve_model(args.model, args.quantization)
    precision = {"fp16": "FP16", "int8": "INT8 (weights-only, W8A16)",
                 "int4": "INT4 (weights-only, W4A16)"}[args.quantization]

    print("========================================================================")
    print(f"         vLLM Chat Application (Qwen2.5 {args.model.upper()})")
    print("========================================================================")
    print(f"[*] Precision : {precision}")
    print(f"[*] Model     : {model_id}")

    llm = LLM(
        model                  = model_id,
        dtype                  = "float16",     # activations + KV cache stay FP16
        max_model_len          = args.max_model_len,
        gpu_memory_utilization = 0.88,
        trust_remote_code      = True,
        enable_prefix_caching  = False,
    )
    sampling = SamplingParams(temperature=0.7, top_p=0.8, top_k=20,
                             repetition_penalty=1.05, max_tokens=args.max_tokens)

    print("\n[✔] Engine Initialized and Ready!")
    print("Type 'exit' or 'quit' to terminate session.")
    print("========================================================================\n")

    conversation = []
    while True:
        try:
            user_input = input("User > ").strip()
        except EOFError:
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        conversation.append({"role": "user", "content": user_input})
        outputs = llm.chat(conversation, sampling, use_tqdm=False)
        reply = outputs[0].outputs[0].text.strip()
        print(f"Qwen > {reply}\n")
        conversation.append({"role": "assistant", "content": reply})

    print("\n[*] Session ended.")


if __name__ == "__main__":
    main()
