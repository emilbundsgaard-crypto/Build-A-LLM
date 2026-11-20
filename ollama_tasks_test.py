"""
ollama_tasks_test.py — tests a local Ollama model 5 times (10 calls: direct + code) on three small tasks:
  1. Which decimal is larger, 9.11 or 9.9?
  2. How many r's are in "strawberry"?
  3. Every third character of "Ubiquitous", starting with the first.

Two conditions per run:
  A) "direct"   — the model answers directly, without code.
  B) "code"     — the model writes Python; this script EXECUTES that code locally
                  (sandboxed: isolated mode, temp folder, 10 s timeout, risky imports blocked)
                  and grades the printed output.

Answers are graded automatically against: 9.9 | 3 | Uqts
Uses only the Python standard library and Ollama's local API (http://localhost:11434).

Run:  python3 ollama_tasks_test.py
      python3 ollama_tasks_test.py --model gemma4:e4b --runs 5
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

TASKS = (
    '1. Which decimal is larger, 9.11 or 9.9?\n'
    '2. How many r\'s are in "strawberry"?\n'
    '3. Extract every third character of "Ubiquitous", starting with the first.'
)

PROMPT_DIRECT = (
    "Answer directly without writing or using any code.\n\n"
    f"{TASKS}\n\n"
    "Reply in exactly this format and nothing else:\n"
    "1: <answer>\n2: <answer>\n3: <answer>"
)

PROMPT_CODE = (
    "Write a short Python program that computes the answers to these three tasks:\n\n"
    f"{TASKS}\n\n"
    "The program must print exactly three lines: line 1 = the larger decimal, "
    "line 2 = the number of r's, line 3 = the extracted characters. "
    "Use only built-in Python (no imports, no files, no network). "
    "Return only one ```python code block."
)

BLOCKED = re.compile(
    r"\b(import|from)\s+(os|sys|subprocess|shutil|socket|pathlib|requests|urllib|ctypes|http)\b"
    r"|__import__|\bopen\s*\(|\beval\s*\(|\bexec\s*\("
)


# ---------------- Ollama ----------------
def api(host, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(host + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(host, model, prompt, seed, temperature):
    options = {"seed": seed}
    if temperature is not None:
        options["temperature"] = temperature
    t0 = time.perf_counter()
    res = api(host, "/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    })
    msg = res.get("message", {})
    return {
        "content": msg.get("content", ""),
        "thinking": msg.get("thinking", ""),
        "seconds": round(time.perf_counter() - t0, 3),
        "eval_count": res.get("eval_count"),
    }


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------- grading ----------------
def grade_q1(ans):
    nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", ans or "")]
    has99 = any(abs(n - 9.9) < 1e-9 for n in nums)
    has911 = any(abs(n - 9.11) < 1e-9 for n in nums)
    return has99 and not has911


def grade_q2(ans):
    ans = (ans or "").lower()
    words = {"three": 3, "two": 2, "four": 4, "one": 1}
    m = re.search(r"\b(\d+|one|two|three|four)\b", ans)
    if not m:
        return False
    v = m.group(1)
    return (int(v) if v.isdigit() else words[v]) == 3


def grade_q3(ans):
    ans = ans or ""
    if re.search(r"\bUqts\b", ans):
        return True
    return re.sub(r"[^A-Za-z]", "", ans) == "Uqts"


def parse_direct(text):
    out = {}
    for k in ("1", "2", "3"):
        m = re.search(rf"^\s*\**{k}\**\s*[:.)]\s*\**\s*(.+?)\s*$", text, flags=re.MULTILINE)
        out[k] = m.group(1).strip() if m else ""
    return out


def extract_code(text):
    m = re.search(r"```(?:python|py)?[^\n]*\n(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def run_code(code, timeout=10):
    if not code:
        return {"status": "no_code", "stdout": "", "stderr": ""}
    if BLOCKED.search(code):
        return {"status": "blocked", "stdout": "", "stderr": "blocked pattern: " + BLOCKED.search(code).group(0)}
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True,
                               text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": ""}
    return {"status": "ok" if p.returncode == 0 else f"error({p.returncode})",
            "stdout": p.stdout, "stderr": p.stderr[-2000:]}


# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma4:e4b")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=None,
                   help="default: the model's own default in Ollama")
    p.add_argument("--seed-start", type=int, default=0, help="run i uses seed seed-start + i")
    p.add_argument("--out-dir", default=str(HERE / "ollama_runs"))
    return p.parse_args()


def main():
    args = parse_args()
    try:
        version = api(args.host, "/api/version", timeout=10).get("version")
        models = [m["name"] for m in api(args.host, "/api/tags", timeout=10).get("models", [])]
    except (urllib.error.URLError, ConnectionError) as e:
        sys.exit(f"ERROR: cannot reach Ollama at {args.host} ({e}). Start the Ollama app and try again.")
    if args.model not in models:
        sys.exit(f"ERROR: model {args.model!r} not found in Ollama. Installed: {models}")

    run_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl = open(run_dir / "results.jsonl", "w", encoding="utf-8")
    rows = []

    print(f"Ollama {version} | model {args.model} | runs {args.runs} | "
          f"temperature {'model default' if args.temperature is None else args.temperature}")
    print(f"Saving to {run_dir.resolve()}\n")
    print(f"{'run':>4} | {'cond':6} | {'Q1 9.9':6} | {'Q2 r=3':6} | {'Q3 Uqts':7} | answers")
    print("-" * 80)

    for i in range(args.runs):
        seed = args.seed_start + i
        for cond, prompt in (("direct", PROMPT_DIRECT), ("code", PROMPT_CODE)):
            try:
                r = chat(args.host, args.model, prompt, seed, args.temperature)
            except Exception as e:
                r = {"content": "", "thinking": "", "seconds": None, "eval_count": None, "error": str(e)}
            content = strip_think(r["content"])
            rec = {"run": i + 1, "seed": seed, "condition": cond, "model": args.model,
                   "prompt": prompt, "raw_response": r["content"], "thinking": r["thinking"],
                   "seconds": r["seconds"], "eval_count": r["eval_count"], "error": r.get("error")}
            if cond == "direct":
                a = parse_direct(content)
                rec["code"], rec["exec"] = None, None
            else:
                code = extract_code(content)
                ex = run_code(code)
                lines = [ln.strip() for ln in ex["stdout"].strip().splitlines()]
                a = {str(k + 1): (lines[k] if k < len(lines) else "") for k in range(3)}
                rec["code"], rec["exec"] = code, ex
            rec["answers"] = a
            rec["correct"] = {"q1": grade_q1(a["1"]), "q2": grade_q2(a["2"]), "q3": grade_q3(a["3"])}
            jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl.flush()
            rows.append(rec)

            c = rec["correct"]
            mark = lambda b: "✓" if b else "✗"
            status = "" if cond == "direct" else f" [{rec['exec']['status']}]"
            print(f"{i + 1:>4} | {cond:6} | {mark(c['q1']):^6} | {mark(c['q2']):^6} | {mark(c['q3']):^7} | "
                  f"{a['1']!r} {a['2']!r} {a['3']!r}{status}")
    jsonl.close()

    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed", "condition", "q1_answer", "q1_correct", "q2_answer", "q2_correct",
                    "q3_answer", "q3_correct", "exec_status", "seconds"])
        for r in rows:
            w.writerow([r["run"], r["seed"], r["condition"],
                        r["answers"]["1"], r["correct"]["q1"], r["answers"]["2"], r["correct"]["q2"],
                        r["answers"]["3"], r["correct"]["q3"],
                        (r["exec"] or {}).get("status", ""), r["seconds"]])

    lines = [f"Model: {args.model} (Ollama {version})",
             f"Runs: {args.runs} | seeds {args.seed_start}..{args.seed_start + args.runs - 1} | "
             f"temperature: {'model default' if args.temperature is None else args.temperature}",
             "Correct answers: Q1 = 9.9 | Q2 = 3 | Q3 = Uqts", ""]
    for cond in ("direct", "code"):
        rs = [r for r in rows if r["condition"] == cond]
        n = len(rs)
        q = {k: sum(r["correct"][k] for r in rs) for k in ("q1", "q2", "q3")}
        allc = sum(all(r["correct"].values()) for r in rs)
        lines.append(f"[{cond}]  Q1 {q['q1']}/{n} | Q2 {q['q2']}/{n} | Q3 {q['q3']}/{n} | all three {allc}/{n}")
        if cond == "code":
            st = {}
            for r in rs:
                s = r["exec"]["status"]
                st[s] = st.get(s, 0) + 1
            lines.append(f"         code execution status: {st}")
    summary = "\n".join(lines) + "\n"
    (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Saved: results.jsonl (all raw responses + code), results.csv, summary.txt in {run_dir.resolve()}")


if __name__ == "__main__":
    main()
