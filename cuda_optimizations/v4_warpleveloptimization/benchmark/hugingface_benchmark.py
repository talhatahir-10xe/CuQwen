import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_huggingface_benchmark():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    print("==================================================================")
    print("   HuggingFace Transformers Benchmark Suite (8000 Tokens)       ")
    print("==================================================================\n")
    print(f"[*] Loading model ({model_id}) in FP16 on GPU...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    print("[✔] Model loaded successfully.\n")

    TOTAL_TOKENS = 8000
    INTERVAL_STEP = 1000

    # Initialize random token sequence matching vocabulary range
    torch.manual_seed(42)
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (1, 1), device=device)

    print("[*] Starting HuggingFace PyTorch benchmark loop (raw model speed)...")

    # Warmup step
    with torch.inference_mode():
        _ = model(dummy_input_ids)
    torch.cuda.synchronize()

    interval_results = []
    past_key_values = None
    input_ids = dummy_input_ids

    total_start_time = time.time()

    for slice_start in range(0, TOTAL_TOKENS, INTERVAL_STEP):
        slice_end = slice_start + INTERVAL_STEP
        
        torch.cuda.synchronize()
        slice_start_time = time.time()

        with torch.inference_mode():
            for _ in range(INTERVAL_STEP):
                outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits
                past_key_values = outputs.past_key_values
                # Greedy choice (argmax) directly on GPU
                input_ids = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

        torch.cuda.synchronize()
        slice_end_time = time.time()

        duration_sec = slice_end_time - slice_start_time
        duration_ms = duration_sec * 1000.0
        tok_per_sec = INTERVAL_STEP / duration_sec

        interval_results.append({
            "start": slice_start + 1,
            "end": slice_end,
            "duration_ms": duration_ms,
            "tok_sec": tok_per_sec
        })

        print(f"  [Progress] Processed tokens {slice_start + 1:4d} to {slice_end:4d} | Speed: {tok_per_sec:.2f} tok/s")

    total_end_time = time.time()
    total_wall_sec = total_end_time - total_start_time

    # Compute Statistics
    avg_tok_sec = sum(r["tok_sec"] for r in interval_results) / len(interval_results)
    initial_speed = interval_results[0]["tok_sec"]
    final_speed = interval_results[-1]["tok_sec"]
    decay_rate = ((initial_speed - final_speed) / initial_speed) * 100.0

    # Print Results Table
    print("\n========================================================================")
    print("              HUGGINGFACE BENCHMARK RESULTS (1.5B-INSTRUCT)             ")
    print("========================================================================")
    print("| Token Range   | Duration (ms) | Speed (tokens/sec) | Context Slice   |")
    print("+---------------+---------------+--------------------+-----------------+")
    for r in interval_results:
        print(f"| {r['start']:5d} - {r['end']:4d} | {r['duration_ms']:13.2f} | {r['tok_sec']:18.2f} | {r['end'] // 1000}k context   |")
    print("+---------------+---------------+--------------------+-----------------+")

    print("\n========================================================================")
    print("                           OVERALL METRICS                              ")
    print("========================================================================")
    print(f"  • Total Generated Tokens   : {TOTAL_TOKENS} tokens")
    print(f"  • Total Time Elapsed       : {total_wall_sec:.2f} seconds")
    print(f"  • Average Speed            : {avg_tok_sec:.2f} tok/s")
    print(f"  • Initial Speed (0-1k)     : {initial_speed:.2f} tok/s")
    print(f"  • Final Speed (7k-8k)      : {final_speed:.2f} tok/s")
    print(f"  • Speed Decay Rate         : {decay_rate:.2f} %")
    print("========================================================================\n")

if __name__ == "__main__":
    run_huggingface_benchmark()