import fire
import logging
import time
from tqdm import tqdm

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from models.compression.monkeypatch import replace_qwen3_5

# from utils.qwen2_norepeat import qwen2_flashattention2_norepeat_forward

# def monkeypatch():
#     transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = qwen2_flashattention2_norepeat_forward

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )

def average_excluding_min_max(numbers):
    if len(numbers) <= 2:
        return sum(numbers) / len(numbers)
    
    numbers_excluding_min_max = numbers.copy()
    numbers_excluding_min_max.remove(min(numbers))
    numbers_excluding_min_max.remove(max(numbers))

    return sum(numbers_excluding_min_max) / len(numbers_excluding_min_max)


def synchronize_all_cuda_devices() -> None:
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_idx)


class FirstTokenTimingCriteria(StoppingCriteria):
    def __init__(self):
        self.first_token_time = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.first_token_time is None:
            synchronize_all_cuda_devices()
            self.first_token_time = time.perf_counter()
        return False


def build_compression_config(compression_mode, compression_budget):
    return {
        "method": compression_mode,
        "method_config": {
            "budget": compression_budget,
            "window_size": 8,
            "mix_lambda": 0.07,
            "retain_ratio": 0.2,
            "retain_direction": "last",
            "first_tokens": 4,
        },
        "compression": None,
        "update_kv": True,
    }


def apply_qwen3_5_compression_setup(model, tokenizer, compression_mode):
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "think",
            "method": compression_mode,
        }
    )
    model.newline_token_ids = [
        tokenizer.encode(text, add_special_tokens=False)[-1]
        for text in ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n"]
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>", add_special_tokens=False)[-1]]


def load_model_and_tokenizer(
    model_path,
    compression=False,
    compression_mode=None,
    compression_budget=4096,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "attn_implementation": "flash_attention_2",
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if compression:
        if not compression_mode:
            raise ValueError("Please provide compression_mode when compression=True.")
        if "qwen3.5" not in model_path.lower():
            raise ValueError(
                f"Compression currently supports only qwen3.5 models, got: {model_path}"
            )

        replace_qwen3_5(build_compression_config(compression_mode, compression_budget))
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        apply_qwen3_5_compression_setup(model, tokenizer, compression_mode)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    model.eval()
    return model, tokenizer


def run_generation_with_timing(model, input_ids, attention_mask, tokenizer, max_new_tokens):
    first_token_timer = FirstTokenTimingCriteria()
    stopping_criteria = StoppingCriteriaList([first_token_timer])

    synchronize_all_cuda_devices()
    start_time = time.perf_counter()
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None,  # Disable EOS token to prevent early termination
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    synchronize_all_cuda_devices()
    total_time = time.perf_counter() - start_time

    generated_tokens = generated_ids.shape[-1] - input_ids.shape[-1]
    if first_token_timer.first_token_time is None:
        ttft = total_time
    else:
        ttft = first_token_timer.first_token_time - start_time

    if generated_tokens <= 1:
        avg_tpot = 0.0
    else:
        avg_tpot = (total_time - ttft) / (generated_tokens - 1)

    return generated_ids, total_time, ttft, avg_tpot, generated_tokens

def measure_throughput(
    model_path: str = "Qwen/QwQ-32B",
    compression: bool = False,
    compression_mode: str = None,
    compression_budget: int = 4096,
    # experiment arguments
    batch_size: int = 16,
    input_len: int = 128,
    output_len: int = 32768,
    num_warmups: int = 1,
    num_runs: int = 3,
    output_file: str = None
    ):

    import datetime
    import os

    # Generate output file name if not provided
    if output_file is None:
        model_name = model_path.split("/")[-1] if "/" in model_path else model_path
        mode_name = compression_mode if compression and compression_mode else "fullkv"
        budget_suffix = f"_budget_{compression_budget}" if compression else ""
        output_file = (
            f"/home/yangx/Qwen3.5_compression/results/efficiency/"
            f"throughput_results_{model_name}_{mode_name}{budget_suffix}_{batch_size}_{input_len}.txt"
        )
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        print("Program terminated to avoid overwriting existing results.")
        return

    num_gpus = torch.cuda.device_count()

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        compression=compression,
        compression_mode=compression_mode,
        compression_budget=compression_budget,
    )

    # Input Sequence      
    input_id = torch.ones((batch_size, input_len), dtype=torch.int64).to(model.device)
    attn_mask = torch.ones((batch_size, input_len), dtype=torch.int64).to(model.device)

    if num_warmups > 0:
        for i in range(num_warmups):
            print(f"Warm Up Run #{i}")

            with torch.no_grad():
                generated_ids, _, _, _, _ = run_generation_with_timing(
                    model=model,
                    input_ids=input_id,
                    attention_mask=attn_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=output_len,
                )

        del generated_ids
        cleanup_memory()
    
    for i in range(num_gpus):
        torch.cuda.reset_peak_memory_stats(device=i)


    results_list = []
    time_list = []  # Store time for each run in seconds
    ttft_list = []  # Store TTFT for each run in seconds
    tpot_list = []  # Store average TPOT for each run in seconds/token

    for i in range(num_runs):
        print(f"Test Run #{i}")

        with torch.no_grad():
            generated_ids, total_time, ttft, avg_tpot, generated_tokens = run_generation_with_timing(
                model=model,
                input_ids=input_id,
                attention_mask=attn_mask,
                tokenizer=tokenizer,
                max_new_tokens=output_len,
            )

        throughput = batch_size * generated_tokens / total_time
        results_list.append(throughput)
        time_list.append(total_time)
        ttft_list.append(ttft)
        tpot_list.append(avg_tpot)

        print(f"Generated IDs length: {generated_ids.shape}")

        del generated_ids
        cleanup_memory()

    avg_throughput = average_excluding_min_max(results_list)
    avg_time = average_excluding_min_max(time_list)  # Average time in seconds
    avg_ttft = average_excluding_min_max(ttft_list)  # Average TTFT in seconds
    avg_tpot = average_excluding_min_max(tpot_list)  # Average TPOT in seconds/token

    total_max_memory = 0
    for i in range(num_gpus):
        max_mem = torch.cuda.max_memory_allocated(device=i)
        total_max_memory += max_mem

    # Prepare results for both console and file output
    results_text = []
    results_text.append("=" * 60)
    results_text.append("THROUGHPUT BENCHMARK RESULTS")
    results_text.append("=" * 60)
    results_text.append(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Model: {model_path}")
    results_text.append(f"Mode: {compression_mode if compression else 'fullkv'}")
    if compression:
        results_text.append(f"Compression Budget: {compression_budget}")
    
    results_text.append(f"Experiment Parameters:")
    results_text.append(f"  Batch Size: {batch_size}")
    results_text.append(f"  Input Length: {input_len}")
    results_text.append(f"  Output Length: {output_len}")
    results_text.append(f"  Number of Warm Up Runs: {num_warmups}")
    results_text.append(f"  Number of Test Runs: {num_runs}")
    results_text.append(f"")
    results_text.append(f"Individual Run Results:")
    for i, (throughput, run_time, ttft, tpot) in enumerate(zip(results_list, time_list, ttft_list, tpot_list)):
        results_text.append(
            f"  Run {i+1}: {throughput:.2f} tokens/sec, {run_time:.2f} seconds, "
            f"TTFT={ttft * 1000:.2f} ms, Avg TPOT={tpot * 1000:.2f} ms/token"
        )
    results_text.append(f"")
    results_text.append(f"Average Throughput (tokens/sec): {avg_throughput:.2f}")
    results_text.append(f"Average Time per Run (seconds): {avg_time:.2f}")
    results_text.append(f"Average TTFT (ms): {avg_ttft * 1000:.2f}")
    results_text.append(f"Average TPOT (ms/token): {avg_tpot * 1000:.2f}")
    results_text.append(f"Peak GPU Memory: {total_max_memory / 1000**2 / 1000:.2f} GB")
    results_text.append("=" * 60)

    # Print to console
    for line in results_text:
        print(line)

    # Save to file
    try:
        with open(output_file, 'w') as f:
            for line in results_text:
                f.write(line + '\n')
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"\nError saving results to file: {e}")

    # Also print the old format for backward compatibility
    print(f"\nModel: {model_path}")
    print(f"Mode: {compression_mode if compression else 'fullkv'}")

    print(f"Batch Size={batch_size}")
    print(f"Input Length={input_len}, Output Length={output_len}")
    print(f"Number of Warm Up Runs={num_warmups}, Number of Test Runs={num_runs}")
    print(f"Average Throughput (tokens/sec)={avg_throughput:.2f}")
    print(f"Average Time per Run (seconds)={avg_time:.2f}")
    print(f"Average TTFT (ms)={avg_ttft * 1000:.2f}")
    print(f"Average TPOT (ms/token)={avg_tpot * 1000:.2f}")
    print(f"Peak GPU Memory: {total_max_memory / 1000**2 / 1000:.2f} GB\n")


if __name__ == "__main__":
    fire.Fire(measure_throughput)
