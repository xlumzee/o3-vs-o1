# parse_and_plot.py
# Usage:
#   python parse_and_plot.py /path/to/raw_log.csv
# The script will:
#   - Parse messy console-like rows (model/task/toks/latency/cost/correct)
#   - Ignore summary blocks
#   - Produce infographics in ./infographics/

import sys
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def normalize_line(ln: str) -> str:
    # Unify whitespace and separators
    ln = ln.replace('\t', ' ')
    # Replace common unicode vertical bars with ASCII pipe
    for ch in ['│', '┃', '┃', '❘', '∣']:
        ln = ln.replace(ch, '|')
    # Collapse multiple spaces
    ln = re.sub(r"\s+", " ", ln).strip()
    return ln

def load_raw_lines(path: Path):
    # Support CSV/TXT; read as raw text lines
    txt = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = txt.splitlines()
    lines = [normalize_line(ln) for ln in raw_lines if normalize_line(ln)]
    return lines

# Regex to capture rows like:
# [o1] logic_1 | 256 toks | 5.9 s | $0.02  ❌
# [o3] code_1 | toks=426 | t=5.85s | cost=$0.00  ✅
ROW_RX = re.compile(
    r"""
    ^\[\s*(?P<model>[^\]]+)\]\s*        # [o1] or [o3-2025-...]
    (?P<task>[a-zA-Z0-9_]+)\s*          # logic_1 / math_1 / code_1
    \|\s*
    (?:
        (?P<tokens1>\d+)\s*toks         # "256 toks"
        |
        toks\s*=\s*(?P<tokens2>\d+)     # or "toks=256"
    )
    \s*\|\s*
    (?:
        t\s*=\s*(?P<lat1>[\d.]+)\s*s    # "t=5.9s"
        |
        (?P<lat2>[\d.]+)\s*s            # or "5.9 s"
    )
    \s*\|\s*
    (?:
        cost\s*=\s*\$?\s*(?P<cost1>[\d.]+)  # "cost=$0.02"
        |
        \$\s*(?P<cost2>[\d.]+)              # or "$0.02"
    )
    \s*
    (?P<correct>[✅❌])
    """,
    re.VERBOSE
)

def parse_rows(lines):
    rows = []
    for ln in lines:
        # Skip obvious non-data blocks
        low = ln.lower()
        if low.startswith("💰") or low.startswith("model") or "avg_latency_s" in low or "avg_cost_usd" in low:
            continue
        if not (']' in ln and '|' in ln):
            continue

        # First try regex
        m = ROW_RX.search(ln)
        if m:
            d = m.groupdict()
            model = d["model"].strip()
            task  = d["task"].strip()
            toks = d["tokens1"] or d["tokens2"]
            tokens = int(toks) if toks and toks.isdigit() else None
            lat = d["lat1"] or d["lat2"]
            latency_s = float(lat) if lat else None
            cost = d["cost1"] or d["cost2"]
            cost_usd = float(cost) if cost else None
            correct = True if d["correct"] == "✅" else False
            rows.append({
                "model": model, "task": task, "tokens": tokens,
                "latency_s": latency_s, "cost_usd": cost_usd, "correct": correct
            })
            continue

        # Fallback: split by pipe and parse segments
        parts = [p.strip() for p in re.split(r"\|", ln) if p.strip()]
        if len(parts) >= 4:
            head, tokseg, latseg, costseg = parts[0], parts[1], parts[2], parts[3]
            # model & task from head
            mm = re.search(r"\[([^\]]+)\]", head)
            if not mm:
                continue
            model = mm.group(1).strip()
            task = re.sub(r"^\[[^\]]+\]", "", head).strip()
            # task is first token-like chunk
            tt = re.search(r"([A-Za-z0-9_]+)", task)
            if not tt:
                continue
            task = tt.group(1)
            # tokens
            mtok = re.search(r"(?:toks\s*=\s*|)(\d+)\s*toks?", tokseg, flags=re.I)
            if not mtok:
                mtok = re.search(r"(\d+)", tokseg)
            tokens = int(mtok.group(1)) if mtok else None
            # latency
            mlat = re.search(r"t\s*=\s*([\d.]+)s", latseg, flags=re.I) or re.search(r"([\d.]+)\s*s", latseg, flags=re.I)
            latency_s = float(mlat.group(1)) if mlat else None
            # cost
            mcost = re.search(r"cost\s*=\s*\$?([\d.]+)", costseg, flags=re.I) or re.search(r"\$\s*([\d.]+)", costseg)
            cost_usd = float(mcost.group(1)) if mcost else None
            # correctness: look anywhere in the line for check/cross
            correct = "✅" in ln

            # Require at least model, task, and one metric
            if model and task and (latency_s is not None or cost_usd is not None or tokens is not None):
                rows.append({
                    "model": model, "task": task, "tokens": tokens,
                    "latency_s": latency_s, "cost_usd": cost_usd, "correct": correct
                })
    return pd.DataFrame(rows)

def model_family(name: str) -> str:
    # Collapse "o1-2024-12-17" -> "o1", "o3-2025-04-16" -> "o3"
    if not isinstance(name, str):
        return ""
    return name.split("-")[0] if "-" in name else name

def make_infographics(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(exist_ok=True)

    # family column
    df["model_family"] = df["model"].map(model_family)

    # ---- Panel 1: Accuracy by Model ----
    acc = (df.groupby("model_family")["correct"]
             .agg(total="count", correct="sum")
             .reset_index())
    plt.figure(figsize=(10.8,10.8))  # 1080x1080 at 100 dpi
    plt.bar(acc["model_family"], acc["correct"])
    for i, r in acc.iterrows():
        label = f"{int(r['correct'])}/{int(r['total'])}"
        plt.text(i, r["correct"] + 0.05, label, ha="center", va="bottom")
    plt.title("Accuracy by Model (tasks correct)")
    plt.ylabel("Correct (count)")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "panel1_accuracy.png", dpi=100)
    plt.close()

    # ---- Panel 2: Latency vs Cost (bubble size = # correct) ----
    perf = (df.groupby("model_family")
              .agg(avg_latency_s=("latency_s","mean"),
                   avg_cost_usd=("cost_usd","mean"),
                   correct=("correct","sum"),
                   total=("correct","count"))
              .reset_index())
    perf["avg_latency_s"] = perf["avg_latency_s"].fillna(0.001)
    perf["avg_cost_usd"]  = perf["avg_cost_usd"].fillna(0.0001)
    sizes = perf["correct"].clip(lower=0)*400 + 200

    plt.figure(figsize=(10.8,10.8))
    plt.scatter(perf["avg_latency_s"], perf["avg_cost_usd"], s=sizes, alpha=0.7)
    for i, r in perf.iterrows():
        txt = f"{r['model_family']}  (cost/run={r['avg_cost_usd']:.4f})"
        plt.text(r["avg_latency_s"]+0.05, r["avg_cost_usd"]+(r["avg_cost_usd"]*0.15+0.0001), txt, fontsize=10)
    plt.xlabel("Avg Latency (s)")
    plt.ylabel("Avg Cost ($)")
    plt.title("Latency vs Cost (Bubble Size = Tasks Correct)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(outdir / "panel2_latency_cost.png", dpi=100)
    plt.close()

    # ---- Panel 3: Cost per Task (side-by-side bars) ----
    cost_task = (df.groupby(["task","model_family"])["cost_usd"]
                   .mean()
                   .reset_index())
    if cost_task.empty:
        plt.figure(figsize=(10.8,10.8))
        plt.text(0.5,0.5,"No cost data to display", ha="center", va="center")
        plt.title("Cost per Task by Model")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outdir / "panel3_cost_per_task.png", dpi=100)
        plt.close()
    else:
        pivot = cost_task.pivot(index="task", columns="model_family", values="cost_usd").fillna(0.0)
        tasks = list(pivot.index)
        x = range(len(tasks))
        plt.figure(figsize=(10.8,10.8))
        bar_width = 0.35
        fams = list(pivot.columns)
        offsets = [(-bar_width/2), (bar_width/2)] if len(fams) >= 2 else [0.0]
        for idx, fam in enumerate(fams):
            vals = pivot[fam].values
            xx = [xi + offsets[idx] for xi in x] if len(fams) >= 2 else x
            plt.bar(xx, vals, width=bar_width, label=fam, alpha=0.85)
        plt.xticks(list(x), tasks)
        plt.ylabel("Avg Cost per Task ($)")
        plt.title("Cost per Task by Model")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        if len(fams) > 1:
            plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "panel3_cost_per_task.png", dpi=100)
        plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_and_plot.py /path/to/raw_log.csv")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lines = load_raw_lines(path)
    df = parse_rows(lines)

    if df.empty:
        print("No structured rows were parsed. Double-check the raw format.")
        print("Tip: Ensure your file contains lines like: [o3] code_1 | 426 toks | 5.85 s | $0.00  ✅")
        print("First 10 raw lines for debugging:")
        for s in lines[:10]:
            print(repr(s))
        sys.exit(2)

    # Diagnostics
    print("Parsed rows:", len(df))
    print(df.head().to_string(index=False))

    out = path.with_suffix(".parsed.csv")
    df.to_csv(out, index=False)
    print("Saved parsed data to:", out)

    make_infographics(df, Path("infographics"))
    print("Saved panels to ./infographics/")

if __name__ == "__main__":
    main()