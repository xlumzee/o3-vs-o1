import pandas as pd
import matplotlib.pyplot as plt

# ---------- Load your experiment data ----------
df = pd.read_csv("merged_experiments.csv")  # or whatever final CSV you saved
df['model'] = df['model'].str.strip()

# ---------- Summary Table ----------
summary = df.groupby("model").agg({
    "avg_latency_s": "mean",
    "avg_cost_usd": "mean",
    "correct_count": "sum"
}).reset_index()

summary["accuracy_%"] = (summary["correct_count"] / (3 * df["run"].nunique())) * 100
print(summary)

# ---------- Latency Comparison ----------
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["avg_latency_s"], label="Avg Latency (s)")
plt.title("Average Latency Comparison")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig("infographic_latency.png", dpi=300)

# ---------- Cost Comparison ----------
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["avg_cost_usd"], color=["#1f77b4", "#ff7f0e"])
plt.title("Average Cost per Query")
plt.ylabel("USD ($)")
plt.tight_layout()
plt.savefig("infographic_cost.png", dpi=300)

# ---------- Accuracy ----------
plt.figure(figsize=(6,4))
plt.bar(summary["model"], summary["accuracy_%"], color=["#2ca02c", "#d62728"])
plt.title("Accuracy Rate (%)")
plt.ylabel("% Correct Tasks")
plt.tight_layout()
plt.savefig("infographic_accuracy.png", dpi=300)

print("✅ Infographics saved as: infographic_latency.png, infographic_cost.png, infographic_accuracy.png")