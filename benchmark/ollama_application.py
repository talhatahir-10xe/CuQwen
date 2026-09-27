"""
Simple Ollama chat application — counterpart to vllm_application.py and
CuQwen's interactive chat. Its purpose is to sanity-check that the FP16 /
INT8 (Q8_0) / INT4 (Q4_0) Ollama models produce coherent output.

Assumes the Ollama server is running and the model tag exists, e.g.:
  qwen2.5:1.5b-fp16 / qwen2.5:1.5b-q8_0 / qwen2.5:1.5b-q4_0
(see benchmark/README.md for the download + `ollama create` commands).

Usage:
  python3 ollama_application.py --model=1.5b --quantization=int8
Type a message at "User >"; type 'exit' or 'quit' (or EOF) to leave.
"""
import sys
import argparse
from ollama import Client

MODEL_SIZES = ["0.5b", "1.5b", "3b", "7b"]
QUANT_SUFFIX = {"fp16": "fp16", "int8": "q8_0", "int4": "q4_0"}


def model_tag(size: str, quant: str) -> str:
    return f"qwen2.5:{size}-{QUANT_SUFFIX[quant]}"


def main():
    parser = argparse.ArgumentParser(description="Ollama interactive chat (FP16/INT8/INT4)")
    parser.add_argument("--model", required=True, choices=MODEL_SIZES,
                        help="Model size variant: 0.5b, 1.5b, 3b, or 7b")
    parser.add_argument("--quantization", default="fp16", choices=["fp16", "int8", "int4"],
                        help="fp16 (default), int8 (Q8_0) or int4 (Q4_0)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per reply")
    args = parser.parse_args()

    tag = model_tag(args.model, args.quantization)
    precision = {"fp16": "FP16", "int8": "INT8 (Q8_0)", "int4": "INT4 (Q4_0)"}[args.quantization]

    print("========================================================================")
    print(f"         Ollama Chat Application (Qwen2.5 {args.model.upper()})")
    print("========================================================================")
    print(f"[*] Precision : {precision} | FP16 KV cache")
    print(f"[*] Model tag : {tag}")

    client = Client()
    options = {"num_gpu": -1, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
               "repeat_penalty": 1.05, "num_predict": args.max_tokens, "f16_kv": True}

    # Fail early with a clear message if the tag is missing.
    try:
        client.show(tag)
    except Exception as e:
        sys.exit(f"[!] Ollama model '{tag}' not found ({e}).\n"
                 f"    Create it first (see benchmark/README.md) — e.g.:\n"
                 f"      ollama create {tag} -f Modelfile")

    print("\n[✔] Engine Ready! Type 'exit' or 'quit' to terminate session.")
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
        resp = client.chat(model=tag, messages=conversation, options=options, stream=False)
        reply = resp["message"]["content"].strip()
        print(f"Qwen > {reply}\n")
        conversation.append({"role": "assistant", "content": reply})

    print("\n[*] Session ended.")


if __name__ == "__main__":
    main()
