"""
train.py — trains a tiny causal transformer on fineweb_corpus.txt for ~60 s on CPU.

Run:  python train.py
All outputs go to runs/<timestamp>/.
"""
import argparse
import codecs
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402

from data_prep import ENCODING_NAME, prepare  # noqa: E402
from model import GPTConfig, TinyGPT  # noqa: E402


class Tee:
    """Writes to the terminal and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="fineweb_corpus.txt")
    p.add_argument("--tokens", default="fineweb_tokens.txt")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--seconds", type=float, default=60.0, help="training time budget (excludes sampling)")
    p.add_argument("--prompt", default="The new product")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--sample-tokens", type=int, default=40)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--threads", type=int, default=0, help="0 = PyTorch default")
    return p.parse_args()


def get_batch(data, block, batch_size, gen):
    ix = torch.randint(len(data) - block, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block] for i in ix])
    return x, y


@torch.no_grad()
def eval_loss(model, data, block, batch=32):
    if data is None:
        return None
    n_win = (len(data) - 1) // block
    if n_win == 0:
        return None
    xs = torch.stack([data[i * block:(i + 1) * block] for i in range(n_win)])
    ys = torch.stack([data[i * block + 1:(i + 1) * block + 1] for i in range(n_win)])
    model.eval()
    total, count = 0.0, 0
    for j in range(0, n_win, batch):
        _, l = model(xs[j:j + batch], ys[j:j + batch])
        total += l.item() * xs[j:j + batch].numel()
        count += xs[j:j + batch].numel()
    model.train()
    return total / count


def stream_sample(model, enc, prompt_text, prompt_ids, id2orig, args, label):
    model.eval()
    gen = torch.Generator().manual_seed(args.seed)  # same seed every time
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    print(f"\n--- sample [{label}] seed={args.seed} temp={args.temperature} top_k={args.top_k} ---")
    sys.stdout.write(prompt_text)
    sys.stdout.flush()
    pieces, gen_ids = [], []
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    for tid in model.generate_stream(idx, args.sample_tokens, args.temperature, args.top_k, gen):
        orig = id2orig[tid]  # internal -> o200k_base ID
        gen_ids.append(orig)
        piece = decoder.decode(enc.decode_single_token_bytes(orig))
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()
            pieces.append(piece)
    tail = decoder.decode(b"", final=True)
    if tail:
        sys.stdout.write(tail)
        pieces.append(tail)
    sys.stdout.write("\n---\n\n")
    sys.stdout.flush()
    model.train()
    return {"label": label, "prompt": prompt_text, "completion": "".join(pieces), "o200k_ids": gen_ids}


def main():
    args = parse_args()
    wall_start = time.perf_counter()
    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_fh = open(run_dir / "console_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cpu"

    # ---- data ----
    data = prepare(args.corpus, args.tokens, out_dir=run_dir)
    enc = tiktoken.get_encoding(ENCODING_NAME)
    id2orig, orig2int = data["id2orig"], data["orig2int"]
    tokens = torch.tensor(data["tokens_internal"], dtype=torch.long)

    split = int(len(tokens) * 0.9)
    train_data, val_data = tokens[:split], tokens[split:]
    block = min(args.block_size, len(train_data) - 2)
    if block < 8:
        sys.exit(f"ERROR: only {len(tokens)} tokens — too few to train.")
    if len(val_data) < block + 1:
        print("[train] validation split too small; val loss disabled.")
        val_data = None

    # ---- prompt ----
    prompt_orig = enc.encode_ordinary(args.prompt)
    if not prompt_orig:
        sys.exit("ERROR: prompt is empty.")
    missing = [t for t in prompt_orig if t not in orig2int]
    if missing:
        names = [enc.decode([t]) for t in missing]
        sys.exit(f"ERROR: prompt tokens not in corpus vocabulary: {names}. Use --prompt with other words.")
    prompt_ids = [orig2int[t] for t in prompt_orig]

    # ---- model ----
    cfg = GPTConfig(vocab_size=len(id2orig), block_size=block, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd)
    model = TinyGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    batch_gen = torch.Generator().manual_seed(args.seed + 1)

    print(f"[train] device={device} threads={torch.get_num_threads()} torch={torch.__version__}")
    print(f"[train] tokens: total={len(tokens):,} train={len(train_data):,} "
          f"val={0 if val_data is None else len(val_data):,}")
    print(f"[train] model: {n_params:,} params | {asdict(cfg)}")
    print(f"[train] uniform-guess loss ln(V) = {math.log(cfg.vocab_size):.4f}")
    print(f"[train] prompt={args.prompt!r} -> {len(prompt_ids)} tokens")

    samples = [stream_sample(model, enc, args.prompt, prompt_ids, id2orig, args, "step 0 (before training)")]
    rows, val_points = [], []

    v0 = eval_loss(model, val_data, block)
    if v0 is not None:
        val_points.append((0, v0))
        print(f"[val] step 0 | val_loss {v0:.4f}")

    print(f"\n{'step':>6} | {'loss':>8} | {'elapsed':>9} | {'tok/s':>9}")
    print("-" * 42)
    train_time, tokens_seen, step, loss_val = 0.0, 0, 0, float("nan")
    try:
        while train_time < args.seconds:
            t0 = time.perf_counter()
            step += 1
            lr = args.lr * min(1.0, step / max(1, args.warmup_steps))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = get_batch(train_data, block, args.batch_size, batch_gen)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_val = loss.item()
            train_time += time.perf_counter() - t0
            tokens_seen += x.numel()
            tps = tokens_seen / train_time
            wall = time.perf_counter() - wall_start
            rows.append({"step": step, "loss": loss_val, "lr": lr, "train_elapsed_s": train_time,
                         "wall_elapsed_s": wall, "tokens_per_s": tps, "val_loss": ""})
            if step % args.log_every == 0:
                print(f"{step:6d} | {loss_val:8.4f} | {train_time:8.2f}s | {tps:9.0f}")
            if step % args.sample_every == 0 and train_time < args.seconds:
                vl = eval_loss(model, val_data, block)
                if vl is not None:
                    val_points.append((step, vl))
                    rows[-1]["val_loss"] = vl
                    print(f"[val] step {step} | val_loss {vl:.4f}")
                samples.append(stream_sample(model, enc, args.prompt, prompt_ids, id2orig, args, f"step {step}"))
    except KeyboardInterrupt:
        print("\n[train] interrupted — saving results.")

    # ---- final ----
    vl = eval_loss(model, val_data, block)
    if vl is not None:
        val_points.append((step, vl))
        if rows:
            rows[-1]["val_loss"] = vl
        print(f"[val] final step {step} | val_loss {vl:.4f}")
    samples.append(stream_sample(model, enc, args.prompt, prompt_ids, id2orig, args, f"final (step {step})"))

    # ---- save ----
    with open(run_dir / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "train_elapsed_s", "wall_elapsed_s",
                                          "tokens_per_s", "val_loss"])
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if rows:
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], lw=1, label="train loss")
    if val_points:
        ax.plot([s for s, _ in val_points], [v for _, v in val_points], "o-", ms=3, label="val loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title(f"TinyGPT on fineweb sample ({train_time:.1f}s CPU training)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss.png", dpi=150)
    plt.close(fig)

    with open(run_dir / "samples.txt", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"=== {s['label']} ===\n{s['prompt']}{s['completion']}\n\n")
    (run_dir / "samples.json").write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")

    torch.save({
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "internal_to_o200k": id2orig,
        "encoding": ENCODING_NAME,
        "step": step,
        "train_seconds": train_time,
        "args": vars(args),
    }, run_dir / "checkpoint.pt")

    wall = time.perf_counter() - wall_start
    info = {
        "data_source": data["source"],
        "validation": data["report"],
        "params": n_params,
        "steps": step,
        "train_seconds": train_time,
        "wall_seconds": wall,
        "tokens_seen": tokens_seen,
        "avg_tokens_per_s": tokens_seen / train_time if train_time else None,
        "final_train_loss": loss_val,
        "final_val_loss": vl,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"[done] steps={step} | train={train_time:.2f}s | wall (incl. sampling)={wall:.2f}s | "
          f"final loss={loss_val:.4f}")
    print(f"[done] saved to {run_dir.resolve()}: console_log.txt, train_log.csv, loss.png, "
          f"samples.txt, samples.json, checkpoint.pt, run_info.json, validation_report.json, vocab_map.json")

    sys.stdout = sys.__stdout__
    log_fh.close()


if __name__ == "__main__":
    main()
