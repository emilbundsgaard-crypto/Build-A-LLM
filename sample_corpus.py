"""
sample_corpus.py — picks a short sentence that appears exactly in fineweb_corpus.txt,
uses its first few words as the prompt, and generates 100 completions with the trained model.

Selection rules:
  1. Split the corpus into sentences (ending in . ! or ?).
  2. Keep short, clean sentences: starts with a capital letter, --min-words to --max-words words,
     only letters/digits/basic punctuation, and found verbatim in the corpus.
  3. Rank by how common the opening word is among all corpus sentences (most common first),
     then shorter sentences first, then order of appearance.
  4. The beginning = the first --words words of the sentence (original spacing kept).
  5. Encode the beginning with o200k_base; if any token is missing from the model's saved
     vocabulary (or the beginning is longer than the context), print why and try the next sentence.

Same model class, tokenizer and sampling settings as train.py (temperature 0.8, top-k 40).
Run:  python sample_corpus.py
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import tiktoken
import torch

from model import GPTConfig, TinyGPT

HERE = Path(__file__).resolve().parent  # Build-A-LLM folder

CLEAN = re.compile(r"^[A-Z][A-Za-z0-9 ,;:'’\-]*[.!?]$")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=str(HERE / "Part 3 - runs" / "20261119_234242"))
    p.add_argument("--corpus", default=str(HERE / "fineweb_corpus.txt"))
    p.add_argument("--words", type=int, default=3, help="number of words in the sentence beginning")
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=15)
    p.add_argument("--n", type=int, default=100, help="number of completions")
    p.add_argument("--seed-start", type=int, default=0, help="completion i uses seed seed-start + i")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="default: sample_tokens from the training run (checkpoint args)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--out", default=str(HERE / "corpus_completions.txt"))
    return p.parse_args()


def load_model(run_dir):
    ckpt_path, vocab_path = run_dir / "checkpoint.pt", run_dir / "vocab_map.json"
    for p in (ckpt_path, vocab_path):
        if not p.is_file():
            sys.exit(f"ERROR: file not found: {p.resolve()}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if vocab.get("encoding") != "o200k_base" or ckpt.get("encoding") != "o200k_base":
        sys.exit(f"ERROR: expected o200k_base, got vocab_map={vocab.get('encoding')!r}, "
                 f"checkpoint={ckpt.get('encoding')!r}")
    id2orig = vocab["internal_to_o200k"]
    if id2orig != ckpt["internal_to_o200k"]:
        sys.exit("ERROR: vocab_map.json does not match the vocabulary stored in checkpoint.pt.")
    cfg = GPTConfig(**ckpt["config"])
    if cfg.vocab_size != len(id2orig):
        sys.exit(f"ERROR: model vocab_size {cfg.vocab_size} != vocab map size {len(id2orig)}")
    model = TinyGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt, id2orig


def candidate_sentences(corpus, args):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", corpus) if s.strip()]
    opening_counts = Counter(s.split()[0] for s in sentences if s.split())
    seen, cands = set(), []
    for order, s in enumerate(sentences):
        n_words = len(s.split())
        if s in seen or not CLEAN.match(s):
            continue
        if not (args.min_words <= n_words <= args.max_words) or n_words <= args.words:
            continue
        if s not in corpus:  # must appear verbatim
            continue
        seen.add(s)
        cands.append((s, opening_counts[s.split()[0]], n_words, order))
    cands.sort(key=lambda c: (-c[1], c[2], c[3]))
    return cands, len(sentences)


def main():
    args = parse_args()
    if args.words < 1:
        sys.exit("ERROR: --words must be at least 1.")
    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        sys.exit(f"ERROR: file not found: {corpus_path.resolve()}")
    corpus = corpus_path.read_text(encoding="utf-8")

    run_dir = Path(args.run_dir)
    model, cfg, ckpt, id2orig = load_model(run_dir)
    orig2int = {o: i for i, o in enumerate(id2orig)}
    enc = tiktoken.get_encoding("o200k_base")

    cands, n_sent = candidate_sentences(corpus, args)
    print(f"Corpus: {n_sent} sentences, {len(cands)} short clean candidates "
          f"({args.min_words}-{args.max_words} words)")
    if not cands:
        sys.exit("ERROR: no candidate sentences found. Try other --min-words/--max-words.")

    chosen = None
    rejected = []
    for s, opening_count, n_words, _ in cands:
        beginning = re.match(r"\S+(?:\s+\S+){%d}" % (args.words - 1), s).group(0)
        ids = enc.encode_ordinary(beginning)
        missing = [t for t in ids if t not in orig2int]
        if missing:
            reason = "missing tokens " + ", ".join(f"{t} {enc.decode([t])!r}" for t in missing)
        elif len(ids) >= cfg.block_size:
            reason = f"beginning has {len(ids)} tokens >= context {cfg.block_size}"
        else:
            chosen = (s, beginning, ids, opening_count)
            break
        rejected.append((s, beginning, reason))
        print(f"[skip] {beginning!r} from {s!r}: {reason}")

    if chosen is None:
        sys.exit(f"ERROR: all {len(cands)} candidates rejected. Nothing generated.")
    sentence, beginning, prompt_orig, opening_count = chosen
    prompt_ids = [orig2int[t] for t in prompt_orig]

    max_new = args.max_new_tokens
    if max_new is None:
        max_new = ckpt.get("args", {}).get("sample_tokens", 40)

    print(f"Chosen sentence: {sentence!r}")
    print(f"Opening word {sentence.split()[0]!r} starts {opening_count} corpus sentences")
    print(f"Beginning: {beginning!r}")
    print("o200k_base tokens: " + ", ".join(f"{t} {enc.decode([t])!r}" for t in prompt_orig))
    print(f"Sampling: temperature={args.temperature} top_k={args.top_k} max_new_tokens={max_new} "
          f"seeds {args.seed_start}..{args.seed_start + args.n - 1}")

    out_path = Path(args.out)
    width = len(str(args.n))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# run_dir: {run_dir}\n")
        f.write(f"# checkpoint step: {ckpt.get('step')}\n")
        f.write(f"# chosen sentence (verbatim in {corpus_path.name}): {sentence}\n")
        f.write(f"# opening word {sentence.split()[0]!r} starts {opening_count} corpus sentences\n")
        f.write(f"# beginning ({args.words} words): {beginning}\n")
        f.write("# o200k_base tokens: " + ", ".join(f"{t} {enc.decode([t])!r}" for t in prompt_orig) + "\n")
        f.write(f"# temperature={args.temperature} top_k={args.top_k} max_new_tokens={max_new}\n")
        f.write(f"# candidates rejected before this one: {len(rejected)}\n")
        for s, b, r in rejected:
            f.write(f"#   {b!r} ({s!r}): {r}\n")
        f.write("\n")
        for i in range(args.n):
            seed = args.seed_start + i
            gen = torch.Generator().manual_seed(seed)
            idx = torch.tensor([prompt_ids], dtype=torch.long)
            new_orig = [id2orig[t] for t in
                        model.generate_stream(idx, max_new, args.temperature, args.top_k, gen)]
            completion = enc.decode(new_orig)  # invalid UTF-8 byte sequences become U+FFFD
            f.write(f"=== {i + 1:0{width}d} | seed {seed} ===\n{beginning}{completion}\n\n")
            print(f"[{i + 1}/{args.n}] seed {seed} done")

    print(f"Saved {args.n} completions to {out_path.resolve()}")


if __name__ == "__main__":
    main()
