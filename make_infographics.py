# make_infographics.py
# Generates three 1080x1080 panels from all_experiments_combined.csv
# Requirements: pandas, matplotlib

import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = "/Users/amulya/Desktop/2025/Coding/StandAlone/o3vo1/Exp_History.xlsx"   # <-- change if needed
OUT_DIR  = Path("infographics")
OUT_DIR.mkdir(exist_ok=True)

# ---------- Load & clean ----------
# Support CSV or Excel; handle encoding edge cases
p = Path(CSV_PATH)
try:
    if p.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(p)
    else:
        # Try UTF-8 first, then fall back gracefully
        try:
            df = pd.read_csv(p, encoding="utf-8", encoding_errors="strict")
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
except UnicodeDecodeError:
    # Last resort fallback
    df = pd.read_csv(p, encoding="latin1")

# Normalize column names if needed
df.columns = [c.strip().lower() for c in df.columns]

# Expected core columns (any extra columns are ignored)
# run_id | model | task | tokens | latency_s | cost_usd | correct | status | error_message
for col in ["model","task","latency_s","cost_usd","correct","status"]:
    if col not in df.columns:
        df[col] = None

# Coerce numerics
for col in ["tokens","latency_s","cost_usd"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Correct → boolean (handles True/False, '✅', '❌', '1', '0', etc.)
def to_bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true","1","yes","y","✅"}:
        return True
    if s in {"false","0","no","n","❌","nan","none",""}:
        return False
    return False

df["correct"] = df["correct"].apply(to_bool)

# Filter to successful rows for metric charts (but keep errors out of metrics)
ok = df.copy()
if "status" in ok.columns:
    ok = ok[ok["status"].fillna("success").str.lower().eq("success")]

# If your models appear as "o1-2024-12-17", group by the prefix "o1"/"o3" for cleaner labels
def model_family(s):
    s = str(s)
    return s.split("-")[0] if "-" in s else s

ok["model_family"] = ok["model"].apply(model_family)

# ---------- Panel 1: Accuracy by model ----------
acc = (ok.groupby("model_family")["correct"]
         .agg(total="count", correct="sum")
         .reset_index())
acc["accuracy"] = acc["correct"]
# Bar chart: number of correct tasks per model
plt.figure(figsize=(10.8,10.8))  # 1080px @ 100dpi
plt.bar(acc["model_family"], acc["accuracy"])
for i, r in acc.iterrows():
    plt.text(i, r["accuracy"] + 0.05, f"{int(r['accuracy'])}/{int(r['total'])}",
             ha="center", va="bottom")
plt.title("Accuracy by Model (tasks correct)")
plt.ylabel("Correct (count)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_DIR / "panel1_accuracy.png", dpi=100)
plt.close()

# ---------- Panel 2: Latency vs Cost (bubble size = # correct) ----------
perf = (ok.groupby("model_family")
          .agg(avg_latency_s=("latency_s","mean"),
               avg_cost_usd=("cost_usd","mean"),
               correct=("correct","sum"),
               total=("correct","count"))
          .reset_index())

# bubble size scaled; avoid 0 size
sizes = perf["correct"].clip(lower=0) * 400 + 200

plt.figure(figsize=(10.8,10.8))
plt.scatter(perf["avg_latency_s"], perf["avg_cost_usd"], s=sizes, alpha=0.7)
for i, r in perf.iterrows():
    label = f"{r['model_family']}  ($/run={r['avg_cost_usd']:.4f})"
    plt.text(r["avg_latency_s"]+0.05, r["avg_cost_usd"]+(r["avg_cost_usd"]*0.05+0.0001), label, fontsize=10)
plt.xlabel("Avg Latency (s)")
plt.ylabel("Avg Cost ($)")
plt.title("Latency vs Cost (Bubble Size = Tasks Correct)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUT_DIR / "panel2_latency_cost.png", dpi=100)
plt.close()

# ---------- Panel 3: Cost per task (side-by-side bars) ----------
cost_task = (ok.groupby(["task","model_family"])["cost_usd"]
               .mean()
               .reset_index())
# Pivot to columns by model for side-by-side bars
pivot = cost_task.pivot(index="task", columns="model_family", values="cost_usd").fillna(0.0)
tasks = list(pivot.index)
x = range(len(tasks))

plt.figure(figsize=(10.8,10.8))
bar_width = 0.35
families = list(pivot.columns)
# If only one family present, still plot one series
offsets = [(-bar_width/2), (bar_width/2)] if len(families) >= 2 else [0.0]

for idx, fam in enumerate(families):
    vals = pivot[fam].values
    xx = [xi + offsets[idx] for xi in x] if len(families) >= 2 else x
    plt.bar(xx, vals, width=bar_width, label=fam, alpha=0.8)

plt.xticks(list(x), tasks)
plt.ylabel("Avg Cost per Task ($)")
plt.title("Cost per Task by Model")
plt.grid(axis="y", linestyle="--", alpha=0.6)
if len(families) > 1:
    plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "panel3_cost_per_task.png", dpi=100)
plt.close()

# ---------- (Optional) print a mini summary for your LinkedIn caption ----------
total_cost = ok["cost_usd"].sum(skipna=True)
total_runs = len(ok)
per_model = (ok.groupby("model_family")
               .agg(total_cost=("cost_usd","sum"),
                    avg_cost=("cost_usd","mean"),
                    avg_latency=("latency_s","mean"),
                    correct=("correct","sum"),
                    total=("correct","count"))
               .reset_index())

print("Saved:")
print(" -", OUT_DIR / "panel1_accuracy.png")
print(" -", OUT_DIR / "panel2_latency_cost.png")
print(" -", OUT_DIR / "panel3_cost_per_task.png")
print("\nQuick summary for your caption:")
print(f"Total runs: {total_runs}, total cost: ${total_cost:.2f}")
for _, r in per_model.iterrows():
    ratio = f"{int(r['correct'])}/{int(r['total'])}"
    print(f"  {r['model_family']}: avg latency {r['avg_latency']:.2f}s, avg cost ${r['avg_cost']:.4f}, accuracy {ratio}")