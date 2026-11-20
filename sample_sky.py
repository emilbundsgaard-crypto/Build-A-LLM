"""
sample_sky.py — loads the trained TinyGPT and generates 100 completions of a prompt.

Uses the same model class (model.py), the o200k_base tokenizer and the same
sampling settings as train.py (temperature 0.8, top-k 40). Each completion uses
its own seed. Results are saved numbered to sky_completions.txt.

Run:  python sample_sky.py
"""
import argparse
import json
import sys
from pathlib import Path

import tiktoken
import torch

from model import GPTConfig, TinyGPT

HERE = Path(__file__).resolve().parent  # Build-A-LLM folder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=str(HERE / "Part 3 - runs" / "20261119_234242"))
    p.add_argument("--prompt", default="The sky is")
    p.add_argument("--n", type=int, default=100, help="number of completions")
    p.add_argument("--seed-start", type=int, default=0, help="completion i uses seed seed-start + i")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="default: sample_tokens from the training run (checkpoint args)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--out", default=str(HERE / "sky_completions.txt"))
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoint.pt"
    vocab_path = run_dir / "vocab_map.json"
    for p in (ckpt_path, vocab_path):
        if not p.is_file():
            sys.exit(f"ERROR: file not found: {p.resolve()}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    encoding_name = vocab.get("encoding")
    if encoding_name != "o200k_base" or ckpt.get("encoding") != "o200k_base":
        sys.exit(f"ERROR: expected o200k_base, got vocab_map={encoding_name!r}, "
                 f"checkpoint={ckpt.get('encoding')!r}")

    id2orig = vocab["internal_to_o200k"]
    if id2orig != ckpt["internal_to_o200k"]:
        sys.exit("ERROR: vocab_map.json does not match the vocabulary stored in checkpoint.pt.")
    orig2int = {o: i for i, o in enumerate(id2orig)}

    cfg = GPTConfig(**ckpt["config"])
    if cfg.vocab_size != len(id2orig):
        sys.exit(f"ERROR: model vocab_size {cfg.vocab_size} != vocab map size {len(id2orig)}")
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    enc = tiktoken.get_encoding("o200k_base")

    # ---- prompt: encode and check every token is in the model's vocabulary ----
    prompt_orig = enc.encode_ordinary(args.prompt)
    if not prompt_orig:
        sys.exit("ERROR: prompt is empty.")
    missing = [(pos, t) for pos, t in enumerate(prompt_orig) if t not in orig2int]
    print(f"Prompt: {args.prompt!r}")
    print("o200k_base tokens: " + ", ".join(f"{t} {enc.decode([t])!r}" for t in prompt_orig))
    if missing:
        lines = [
            f"# run_dir: {run_dir}",
            f"# checkpoint step: {ckpt.get('step')}",
            f"# prompt: {args.prompt!r}",
            "# o200k_base tokens: " + ", ".join(f"{t} {enc.decode([t])!r}" for t in prompt_orig),
            f"# model vocabulary size: {len(id2orig):,} (only token IDs observed in the training corpus)",
            "",
            f"RESULT: 0 of {args.n} completions generated.",
            f"{len(missing)} of {len(prompt_orig)} prompt tokens are not in the model's vocabulary:",
        ]
        lines += [f"  position {pos}: id {t} = {enc.decode([t])!r}" for pos, t in missing]
        lines.append("No substitution was made, so the model cannot be given this prompt.")
        text = "\n".join(lines) + "\n"
        Path(args.out).write_text(text, encoding="utf-8")
        print("ERROR:", text.split("RESULT: ", 1)[1], end="")
        print(f"Saved this result to {Path(args.out).resolve()}")
        sys.exit(1)
    prompt_ids = [orig2int[t] for t in prompt_orig]

    max_new = args.max_new_tokens
    if max_new is None:
        max_new = ckpt.get("args", {}).get("sample_tokens", 40)

    print(f"Model: step {ckpt.get('step')} | {sum(p.numel() for p in model.parameters()):,} params | "
          f"vocab {cfg.vocab_size:,} | block {cfg.block_size}")
    print(f"Sampling: temperature={args.temperature} top_k={args.top_k} max_new_tokens={max_new} "
          f"seeds {args.seed_start}..{args.seed_start + args.n - 1}")

    out_path = Path(args.out)
    width = len(str(args.n))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# run_dir: {run_dir}\n")
        f.write(f"# checkpoint step: {ckpt.get('step')}\n")
        f.write(f"# prompt: {args.prompt!r}\n")
        f.write(f"# temperature={args.temperature} top_k={args.top_k} max_new_tokens={max_new}\n\n")
        for i in range(args.n):
            seed = args.seed_start + i
            gen = torch.Generator().manual_seed(seed)
            idx = torch.tensor([prompt_ids], dtype=torch.long)
            new_orig = [id2orig[t] for t in
                        model.generate_stream(idx, max_new, args.temperature, args.top_k, gen)]
            completion = enc.decode(new_orig)  # invalid UTF-8 byte sequences become U+FFFD
            f.write(f"=== {i + 1:0{width}d} | seed {seed} ===\n{args.prompt}{completion}\n\n")
            print(f"[{i + 1}/{args.n}] seed {seed} done")

    print(f"Saved {args.n} completions to {out_path.resolve()}")


if __name__ == "__main__":
    main()
