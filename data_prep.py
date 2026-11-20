"""
data_prep.py

Loads fineweb_corpus.txt and fineweb_tokens.txt, validates the token IDs against
GPT-4o's o200k_base tokenizer, regenerates them from the corpus if they do not
match, and builds a compact internal vocabulary (observed IDs only).

Standalone:   python data_prep.py
train.py calls prepare() automatically, so this step is optional.
"""
import json
import re
import sys
from pathlib import Path

import tiktoken

ENCODING_NAME = "o200k_base"  # GPT-4o tokenizer


def _load_token_file(path):
    raw = Path(path).read_text(encoding="utf-8")
    ids = [int(t) for t in re.findall(r"\d+", raw)]
    junk = sorted(set(re.sub(r"[0-9,\s\[\]]", "", raw)))
    return ids, junk


def _first_mismatch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _safe_decode(enc, ids):
    try:
        return enc.decode(ids)
    except Exception as e:
        return f"<could not decode: {e}>"


def prepare(corpus_path="fineweb_corpus.txt", tokens_path="fineweb_tokens.txt", out_dir="prepared"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (corpus_path, tokens_path):
        if not Path(p).is_file():
            sys.exit(f"ERROR: file not found: {Path(p).resolve()}")

    enc = tiktoken.get_encoding(ENCODING_NAME)
    corpus = Path(corpus_path).read_text(encoding="utf-8")
    file_ids, junk = _load_token_file(tokens_path)

    print(f"[data] tokenizer: {ENCODING_NAME} (max token id {enc.max_token_value})")
    print(f"[data] corpus: {len(corpus):,} characters")
    print(f"[data] token file: {len(file_ids):,} IDs")
    if junk:
        print(f"[data] WARNING: non-numeric characters in token file were ignored: {junk}")

    report = {
        "encoding": ENCODING_NAME,
        "corpus_chars": len(corpus),
        "token_file_ids": len(file_ids),
        "token_file_junk_chars": junk,
    }

    # Candidates: exact corpus, and corpus without trailing newline(s)
    candidates = [("exact", corpus)]
    if corpus.rstrip("\r\n") != corpus:
        candidates.append(("trailing_newlines_stripped", corpus.rstrip("\r\n")))

    matched = None
    for label, text in candidates:
        if enc.encode_ordinary(text) == file_ids:
            matched = (label, text)
            break

    if matched:
        label, text = matched
        ids = file_ids
        source = "token_file"
        print(f"[data] VALIDATION PASSED: token file == o200k_base encoding of corpus ({label}).")
        report.update(validation="passed", matched_variant=label)
    else:
        text = corpus
        ref = enc.encode_ordinary(corpus)
        invalid = [t for t in file_ids if t > enc.max_token_value]
        idx = _first_mismatch(file_ids, ref)
        print("[data] VALIDATION FAILED: token file does NOT match the corpus.")
        print(f"[data]   re-encoded corpus length: {len(ref):,} | token file length: {len(file_ids):,}")
        print(f"[data]   IDs above max token value: {len(invalid)}")
        if idx is not None:
            lo, hi = max(0, idx - 8), idx + 8
            print(f"[data]   first mismatch at index {idx}")
            if not invalid:
                print(f"[data]   token file : {_safe_decode(enc, file_ids[lo:hi])!r}")
            print(f"[data]   corpus     : {enc.decode(ref[lo:hi])!r}")
        ids = ref
        source = "regenerated_from_corpus"
        regen_path = out_dir / "fineweb_tokens_regenerated.txt"
        regen_path.write_text(", ".join(map(str, ids)), encoding="utf-8")
        print(f"[data] Regenerated {len(ids):,} IDs from corpus -> {regen_path}")
        report.update(
            validation="failed",
            first_mismatch_index=idx,
            invalid_ids=len(invalid),
            regenerated_ids=len(ids),
            regenerated_file=str(regen_path),
        )

    if enc.decode(ids) != text:
        sys.exit("ERROR: decoded tokens do not reproduce the corpus text. Stopping.")
    print("[data] round-trip decode check passed.")

    # Compact vocabulary: internal index <-> o200k_base ID
    id2orig = sorted(set(ids))
    orig2int = {o: i for i, o in enumerate(id2orig)}
    internal = [orig2int[t] for t in ids]
    print(f"[data] compact vocab: {len(id2orig):,} observed IDs (of {enc.max_token_value + 1:,})")

    report.update(source=source, n_tokens=len(ids), compact_vocab_size=len(id2orig))
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "vocab_map.json").write_text(
        json.dumps({"encoding": ENCODING_NAME, "internal_to_o200k": id2orig}), encoding="utf-8"
    )

    return {
        "source": source,
        "id2orig": id2orig,
        "orig2int": orig2int,
        "tokens_internal": internal,
        "report": report,
    }


if __name__ == "__main__":
    prepare()
