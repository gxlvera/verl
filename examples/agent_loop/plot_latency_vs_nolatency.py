#!/usr/bin/env python3
"""No-shadow bs256: no-latency vs serpapi-latency.

Fig 1 (throughput): overall tok/s bar + gen_throughput over time.
Fig 2 (sglang): running / queue / KV usage over time.
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
    ("no tool latency", "think_8192_500pt_4tool_20260608_111923", "tab:green", 6325.2),
    ("serpapi tool latency (~2.7s)", "think_8192_500pt_serpapilat_20260608_144554", "tab:purple", 4923.7),
]


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


def load(tag):
    fp = glob.glob(f"{LOG}/baseline_*{tag}_bs256*_sglang_metrics.jsonl")[0]
    run, que, tu, gt = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
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
                elif kk == "sglang:gen_throughput":
                    gt[t].append(v or 0.0)
    ts = sorted(run)
    run_s = [sum(run[t]) for t in ts]
    que_s = [sum(que[t]) for t in ts]
    tu_a = [statistics.mean(tu[t]) if tu[t] else 0.0 for t in ts]
    gt_s = [sum(gt[t]) if gt[t] else 0.0 for t in ts]
    idx = [i for i, r in enumerate(run_s) if r >= 1]
    i0, i1 = idx[0], idx[-1]
    sl = slice(i0, i1 + 1)
    ts, run_s, que_s, tu_a, gt_s = ts[sl], run_s[sl], que_s[sl], tu_a[sl], gt_s[sl]
    t0 = ts[0]
    return [t - t0 for t in ts], run_s, que_s, tu_a, gt_s


data = {label: (load(tag) + (color, ov)) for label, tag, color, ov in RUNS}

# ---------------- Figure 1: throughput ----------------
fig1, (axb, axg) = plt.subplots(1, 2, figsize=(14, 5))
labels = list(data.keys())
ovs = [data[l][6] for l in labels]
colors = [data[l][5] for l in labels]
bars = axb.bar(labels, ovs, color=colors)
axb.set_ylabel("overall throughput (tok/s)")
axb.set_title("Overall rollout throughput (perf/throughput)")
for b, v in zip(bars, ovs):
    axb.text(b.get_x() + b.get_width() / 2, v + 50, f"{v:.0f}", ha="center", fontsize=11)
axb.grid(alpha=0.3, axis="y")

for label, (T, run_s, que_s, tu_a, gt_s, color, ov) in data.items():
    axg.plot(T, gt_s, color=color, label=label, lw=1.5)
axg.set_ylabel("gen_throughput (tok/s, sum over servers)")
axg.set_xlabel("time since first sample (s)")
axg.set_title("SGLang generation throughput over time")
axg.legend(loc="upper right", fontsize=9)
axg.grid(alpha=0.3)
fig1.tight_layout()
out1 = f"{LOG}/throughput_latency_vs_nolatency_bs256.png"
fig1.savefig(out1, dpi=120)
print("saved:", out1)

# ---------------- Figure 2: sglang ----------------
fig2, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for label, (T, run_s, que_s, tu_a, gt_s, color, ov) in data.items():
    axes[0].plot(T, run_s, color=color, label=f"{label} (makespan={T[-1]:.0f}s)", lw=1.6)
    axes[1].plot(T, que_s, color=color, label=label, lw=1.6)
    axes[2].plot(T, tu_a, color=color, label=label, lw=1.6)
axes[0].set_title("Running requests (num_running_reqs)")
axes[0].set_ylabel("requests")
axes[1].set_title("Queued requests (num_queue_reqs)")
axes[1].set_ylabel("requests")
axes[2].set_title("KV cache utilization (token_usage, avg/server)")
axes[2].set_ylabel("KV util")
axes[2].axhline(1.0, color="gray", ls=":", alpha=0.6)
axes[2].set_ylim(0, 1.02)
axes[2].set_xlabel("time since first sample (s)")
for ax in axes:
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
fig2.tight_layout()
out2 = f"{LOG}/sglang_latency_vs_nolatency_bs256.png"
fig2.savefig(out2, dpi=120)
print("saved:", out2)

for label, (T, run_s, que_s, tu_a, gt_s, color, ov) in data.items():
    print(f"{label}: tp={ov:.0f} makespan={T[-1]:.1f}s peakKV={max(tu_a):.3f} peakQueue={max(que_s):.0f}")
