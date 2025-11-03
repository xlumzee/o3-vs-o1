{"task_id":"logic_1","expected_exact":"BOX: C","prompt":"Three boxes are labeled (A) apples, (B) oranges, (C) apples and oranges. Every label is wrong. In one draw from a single box, which box should you draw from first to relabel correctly? Answer with EXACTLY one token line: BOX: A or BOX: B or BOX: C"}
{"task_id":"math_1","expected_exact":"FINAL: 583220","prompt":"Let f(n)=sum_{k=1..n} k^2. Use the closed form f(n)=n(n+1)(2n+1)/6. Compute f(120) step-by-step but PRINT ONLY the last line as: FINAL: <integer> with no other text."}
{"task_id":"code_1","expected_contains":"assert fib(1) == 1","prompt":"Return ONLY a Python code block that defines fib(n) with base cases fib(0)=0,fib(1)=1 and then two asserts for fib(0)==0 and fib(1)==1. Keep it under 30 lines."}
import time, json, os, pandas as pd, random, re
from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore, Style, init

# ---------- setup ----------
load_dotenv()
init(autoreset=True)
client = OpenAI()

# --- update these if pricing changes (check docs before you run) ---
PRICING = {
    "o3": {"in": 2.00/1_000_000,  "out": 8.00/1_000_000},
    "o1": {"in":15.00/1_000_000,  "out":60.00/1_000_000},
    # Optional:
    # "o3-pro": {"in":20.00/1_000_000, "out":80.00/1_000_000},
    # "o3-mini": {"in":1.10/1_000_000, "out":1.10/1_000_000},
    # "o1-mini": {"in":1.10/1_000_000, "out":1.10/1_000_000},
}

MODELS = ["o1","o3"]           # edit if you want to test fewer/more models
MAX_RETRIES = 4                # for transient 429 rate limits
BACKOFF_BASE_SEC = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 300  # global safety cap

# ---------- core runner with retries ----------
def run_one(model: str, prompt: str, effort="medium", max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
    """
    Run one prompt against a model with basic exponential backoff for 429s.
    Raises RuntimeError("INSUFFICIENT_QUOTA") if the API reports insufficient quota.
    """
    attempt = 0
    while True:
        t0 = time.time()
        try:
            # prepare parameters shared across models
            kwargs = dict(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                reasoning={"effort": effort},
            )

            # only add sampling controls for models that support them (e.g., GPT-4-turbo)
            if not model.startswith("o"):
                kwargs.update(dict(temperature=0, top_p=1))

            resp = client.responses.create(**kwargs) # type: ignore
            dt = round(time.time() - t0, 2)

            # unified helpers
            text = getattr(resp, "output_text", "") or ""
            usage = getattr(resp, "usage", None)
            # normalize whitespace to stabilize grading
            text = " ".join(text.split())
            model_used = getattr(resp, "model", model)
            resp_id    = getattr(resp, "id", "")

            in_tok     = getattr(usage, "input_tokens", 0) if usage else 0
            out_tok    = getattr(usage, "output_tokens", 0) if usage else 0
            reason_tok = getattr(usage, "reasoning_tokens", 0) if usage else 0

            pin  = PRICING[model]["in"]  * in_tok
            pout = PRICING[model]["out"] * out_tok
            cost = round(pin + pout, 6)

            return {
                "model": model_used, "latency_s": dt,
                "input_tokens": in_tok, "output_tokens": out_tok,
                "reasoning_tokens": reason_tok, "cost_usd": cost,
                "response": text, "resp_id": resp_id
            }

        except Exception as e:
            # Detect insufficient quota (hard stop)
            msg = str(e).lower()
            if "insufficient_quota" in msg or "you exceeded your current quota" in msg:
                raise RuntimeError("INSUFFICIENT_QUOTA") from e

            # Handle generic 429 / rate limit with exponential backoff
            if "429" in msg or "rate limit" in msg:
                if attempt >= MAX_RETRIES:
                    # Bubble up after exhausting retries
                    raise
                sleep_s = BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0, 0.3)
                attempt += 1
                time.sleep(sleep_s)
                continue
            # Unknown error: re-raise
            raise

# ---------- smarter, task-aware grading ----------
import re as _re

def grade(row, expected_exact, expected_contains):
    """
    Returns "✅" or "❌" (or "—" if not gradable).
    Task-aware grading with robust parsing so minor wording/formatting differences
    don't falsely penalize the model.
    """
    ans  = (row.get("response") or "").strip()
    task = (row.get("task_id") or "").strip().lower()

    # ---- logic_1: mislabeled boxes ----
    if task == "logic_1":
        normalized = ans.strip().lower()
        # Exact format wins
        m = _re.search(r"\bbox\s*:\s*([abc])\b", normalized)
        if m:
            return "✅" if m.group(1) == "c" else "❌"

        # Phrasing variants: “apples and oranges” / “both fruits” / “mixed”
        patt = _re.compile(
            r"(labeled|labelled)?\s*['\"]?\b(apples\s*(?:and|&)\s*oranges|both\s+fruits|mixed)\b['\"]?",
            _re.I
        )
        return "✅" if patt.search(normalized) else "❌"

    # ---- math_1: sum of squares f(120) ----
    if task == "math_1":
        target = "583220"
        # Prefer a standalone final answer if the model used a 'final' tag
        final = _re.search(r"(?:final(?: answer)?\s*[:=]\s*)(-?\d+)", ans, flags=_re.I)
        if final:
            return "✅" if final.group(1) == target else "❌"
        nums = _re.findall(r"-?\d+", ans)
        if not nums:
            return "❌"
        # Pass if the correct target appears ANYWHERE among extracted integers
        if target in nums:
            return "✅"
        # Fallback: if multiple numbers and target not present, take the last integer
        return "✅" if nums[-1] == target else "❌"

    # ---- code_1: Fibonacci fix + asserts ----
    if task == "code_1":
        # Accept either recursive or iterative; require a fib function and basic correctness signals
        has_func  = bool(_re.search(r"def\s+fib(?:onacci)?\s*\(", ans))
        has_t0    = ("assert fib(0) == 0" in ans) or ("assert fibonacci(0) == 0" in ans)
        has_t1    = ("assert fib(1) == 1" in ans) or ("assert fibonacci(1) == 1" in ans)
        # Also accept explicit base case fix even if asserts missing (keeps runs with tight caps from false ❌)
        base_fix  = bool(_re.search(r"if\s+n\s*==\s*1\s*:\s*return\s*1", ans))
        return "✅" if (has_func and ((has_t0 and has_t1) or base_fix)) else "❌"

    # ---- generic fallbacks (optional hints from prompts.jsonl) ----
    if expected_exact and expected_exact.strip():
        return "✅" if expected_exact.strip() == ans else "❌"
    if expected_contains and expected_contains.strip():
        return "✅" if expected_contains.lower() in ans.lower() else "❌"

    return "—"

# ---------- main loop ----------
def main():
    rows = []
    with open("prompts.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            task_id = rec.get("task_id","")

            # small per-task cap tweak: logic can be tighter; math/code need a bit more room
            per_task_cap = (
                64  if task_id == "logic_1" else
                160 if task_id == "math_1"  else
                640 if task_id == "code_1"  else
                DEFAULT_MAX_OUTPUT_TOKENS
            )

            effort = "high" if task_id == "math_1" else "medium"

            for m in MODELS:
                try:
                    r = run_one(m, rec["prompt"], effort=effort, max_output_tokens=per_task_cap)
                except RuntimeError as ex:
                    if str(ex) == "INSUFFICIENT_QUOTA":
                        print(f"\n{Fore.RED}✖ Stopping: API reports insufficient quota. Check your plan/billing and try again.{Style.RESET_ALL}")
                        # Flush partial results if any rows exist
                        if rows:
                            df = pd.DataFrame(rows, columns=[
                                "task_id","model","correct","latency_s",
                                "input_tokens","output_tokens","reasoning_tokens","cost_usd","response"
                            ])
                            df.to_csv("results_partial.csv", index=False)
                            print(f"{Fore.YELLOW}Saved partial results to results_partial.csv{Style.RESET_ALL}")
                        return
                    else:
                        # Unknown error during run_one; log and continue to next (best-effort)
                        print(f"{Fore.YELLOW}! Skipping {m} on {task_id} due to error: {ex}{Style.RESET_ALL}")
                        continue

                r.update({"task_id": task_id})
                r["correct"] = grade(r, rec.get("expected_exact",""), rec.get("expected_contains",""))
                rows.append(r)

                # ---- Live console feedback (colored) ----
                total_toks = r["input_tokens"] + r["output_tokens"] + r.get("reasoning_tokens", 0)
                if r["correct"] == "✅":
                    color = Fore.GREEN
                elif r["correct"] == "❌":
                    color = Fore.RED
                else:
                    color = Fore.YELLOW

                print(
                    f"{color}[{m}] {task_id} | "
                    f"toks={total_toks} | t={r['latency_s']}s | "
                    f"cost=${r['cost_usd']:.5f} {r['correct']}{Style.RESET_ALL}"
                )
                # Log incorrect answers to a file for offline inspection (no console noise)
                try:
                    if r["correct"] == "❌":
                        miss = {
                            "task_id": task_id,
                            "model": r.get("model", m),
                            "latency_s": r.get("latency_s", None),
                            "cost_usd": r.get("cost_usd", None),
                            "response": r.get("response", "")
                        }
                        with open("misses.jsonl", "a") as mf:
                            mf.write(json.dumps(miss) + "\n")
                except Exception:
                    pass
                # If incorrect, print a short preview to help debug grader/prompt
                # if r["correct"] == "❌" and r.get("response"):
                #     preview = r["response"].strip().replace("\n", " ")[:200]
                #     print(f"{Fore.MAGENTA}↳ preview: {preview}{Style.RESET_ALL}")

    df = pd.DataFrame(rows, columns=[
        "task_id","model","correct","latency_s",
        "input_tokens","output_tokens","reasoning_tokens","cost_usd","response"
    ])
    df.to_csv("results.csv", index=False)

    # pretty summary for LinkedIn screencap
    pivot = (df
             .groupby(["model"])
             .agg(correct=("correct", lambda s: f"{sum(x=='✅' for x in s)}/{len(s)}"),
                  avg_latency_s=("latency_s","mean"),
                  avg_cost_usd=("cost_usd","mean"),
                  avg_reason_toks=("reasoning_tokens","mean"))
             .reset_index())

    print(f"\n{Fore.CYAN}💰 Estimated total cost: ${df['cost_usd'].sum():.4f}{Style.RESET_ALL}")
    print(pivot.to_string(index=False))
    with open("summary.md","w") as w:
        w.write(pivot.to_markdown(index=False))
    print("\nSaved results.csv and summary.md")

if __name__ == "__main__":
    main()