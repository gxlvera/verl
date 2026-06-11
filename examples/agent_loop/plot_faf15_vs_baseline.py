#!/usr/bin/env python3
"""4-panel SGLang overlay: total requests / running / queue / KV usage.

Compares the tail-15% + 30%-hit shadow run (fire-and-forget) against the
no-shadow baseline, both with serpapi tool latency, bs256.
"""
import glob
import json
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/tiger/verl/logs"
RUNS = [
    ("no-shadow baseline (serpapi)", "think_8192_500pt_serpapilat_20260608_144554", "tab:gray"),
    ("tail-15% shadow + 30% hit (faf)", "think_8192_500pt_faf15_20260608_171456", "tab:red"),
]


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


def load(tag):
    fp = glob.glob(f"{LOG}/baseline_*{tag}_bs256*_sglang_metrics.jsonl")[0]
    run, que, tu = defaultdict(list), defaultdict(list), defaultdict(list)
    with open(fp, errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "series" not in d:
                continue
            t = round(d["monotonic_s"], 1)
            for k, v in d["series"].items():
                kk = strip(k)
                if kk == "sglang:num_running_reqs":
                    run[t].append(v or 0.0)
                elif kk == "sglang:num_queue_reqs":
                    que[t].append(v or 0.0)
                elif kk == "sglang:token_usage":
                    tu[t].append(v or 0.0)
    ts = sorted(run)
    run_s = [sum(run[t]) for t in ts]
    que_s = [sum(que[t]) for t in ts]
    tot_s = [r + q for r, q in zip(run_s, que_s)]
    tu_a = [statistics.mean(tu[t]) if tu[t] else 0.0 for t in ts]
    idx = [i for i, r in enumerate(run_s) if r >= 1]
    i0, i1 = idx[0], idx[-1]
    sl = slice(i0, i1 + 1)
    ts, run_s, que_s, tot_s, tu_a = ts[sl], run_s[sl], que_s[sl], tot_s[sl], tu_a[sl]
    t0 = ts[0]
    T = [t - t0 for t in ts]
    return T, tot_s, run_s, que_s, tu_a


fig, axes = plt.subplots(4, 1, figsize=(13, 15), sharex=True)
titles = [
    "Total requests in system (running + queue)",
    "Running requests (num_running_reqs)",
    "Queued requests (num_queue_reqs)",
    "KV cache utilization (token_usage, avg/server)",
]
ylabels = ["requests", "requests", "requests", "KV util"]

stats = {}
for label, tag, color in RUNS:
    T, tot_s, run_s, que_s, tu_a = load(tag)
    stats[label] = (T[-1], max(tot_s), max(run_s), max(que_s), max(tu_a))
    for ax, ydata in zip(axes, [tot_s, run_s, que_s, tu_a]):
        ax.plot(T, ydata, color=color, label=label, lw=1.6)

for ax, title, yl in zip(axes, titles, ylabels):
    ax.set_title(title)
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
axes[3].axhline(1.0, color="gray", ls=":", alpha=0.6)
axes[3].set_ylim(0, 1.02)
axes[3].set_xlabel("time since first sample (s)")

fig.tight_layout()
out = f"{LOG}/sglang_faf15_vs_baseline_bs256.png"
fig.savefig(out, dpi=120)
print("saved:", out)
for k, v in stats.items():
    print(f"{k}: makespan={v[0]:.1f}s peakTotal={v[1]:.0f} peakRun={v[2]:.0f} peakQueue={v[3]:.0f} peakKV={v[4]:.3f}")
