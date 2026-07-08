"""Measure policy-model decode throughput to feed the GRPO step-time estimate.

Reports single-stream decode tok/s and batched aggregate tok/s at a target
concurrency. Backend is HuggingFace transformers (no vLLM/sglang dependency) --
so these are a LOWER BOUND on rollout throughput: verl's sglang rollout adds
paged attention + continuous batching and will be meaningfully faster,
especially at high concurrency. Still the right shape for sanity-checking
tokens/step math on the actual GPU.

    python -m training.throughput_bench --model Qwen/Qwen3-0.6B --concurrency 64
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--new-tokens", type=int, default=256)
    p.add_argument("--prompt-tokens", type=int, default=512, help="approx prefill length")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()

    # A realistic-ish prompt padded to ~prompt_tokens so prefill cost is counted.
    base = "You are an entity-retrieval agent. " * 40
    prompt = tok.decode(tok(base)["input_ids"][: args.prompt_tokens], skip_special_tokens=True)

    def run_batch(n: int) -> tuple[float, float, float]:
        prompts = [f"{prompt}\nQuestion {i}: name a relevant entity." for i in range(n)]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        torch.cuda.synchronize() if dev == "cuda" else None
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
            )
        torch.cuda.synchronize() if dev == "cuda" else None
        dt = time.time() - t0
        gen = int((out.shape[1] - enc["input_ids"].shape[1]) * n)
        return dt, gen, gen / dt

    # warmup (kernels, cudnn autotune)
    run_batch(1)
    run_batch(min(8, args.concurrency))

    single_dt, single_gen, single_tps = run_batch(1)
    batch_dt, batch_gen, batch_tps = run_batch(args.concurrency)

    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0.0
    result = {
        "model": args.model,
        "device": dev,
        "dtype": str(dtype).replace("torch.", ""),
        "backend": "transformers",
        "prompt_tokens": args.prompt_tokens,
        "new_tokens": args.new_tokens,
        "single_stream": {
            "decode_tok_s": round(single_tps, 1),
            "wall_s": round(single_dt, 2),
        },
        "batched": {
            "concurrency": args.concurrency,
            "aggregate_tok_s": round(batch_tps, 1),
            "per_stream_tok_s": round(batch_tps / args.concurrency, 1),
            "wall_s": round(batch_dt, 2),
        },
        "speedup_batched_vs_single": round(batch_tps / single_tps, 1),
        "peak_vram_gb": round(mem_gb, 2),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
