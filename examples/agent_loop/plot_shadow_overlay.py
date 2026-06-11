#!/usr/bin/env python3
"""Overlay SGLang metrics (running reqs, queue reqs, KV usage) for several runs.

Usage:
    python plot_shadow_overlay.py OUT.png "label1=TAG1" "label2=TAG2" ...
Each TAG matches logs/baseline_*{TAG}_bs256*_sglang_metrics.jsonl
"""
import glob
import json
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/tiger/verl/logs"


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


def load(tag):
    fps = glob.glob(f"{LOG}/baseline_*{tag}_bs256*_sglang_metrics.jsonl")
    if not fps:
        raise FileNotFoundError(tag)
    fp = fps[0]
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
    tu_a = [statistics.mean(tu[t]) if tu[t] else 0.0 for t in ts]
    idx = [i for i, r in enumerate(run_s) if r >= 1]
    i0, i1 = idx[0], idx[-1]
    ts, run_s, que_s, tu_a = ts[i0 : i1 + 1], run_s[i0 : i1 + 1], que_s[i0 : i1 + 1], tu_a[i0 : i1 + 1]
    t0 = ts[0]
    T = [t - t0 for t in ts]
    return T, run_s, que_s, tu_a


out = sys.argv[1]
specs = [a.split("=", 1) for a in sys.argv[2:]]
colors = ["tab:gray", "tab:red", "tab:orange", "tab:green", "tab:blue", "tab:purple"]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
for i, (label, tag) in enumerate(specs):
    T, run_s, que_s, tu_a = load(tag)
    c = colors[i % len(colors)]
    ax1.plot(T, run_s, color=c, label=f"{label} (end={T[-1]:.0f}s, peakR={max(run_s):.0f})")
    ax2.plot(T, que_s, color=c, label=label)
    ax3.plot(T, tu_a, color=c, label=label)

ax1.set_ylabel("running reqs (sum)")
ax1.set_title("SGLang running requests over time (bs256, serpapi tool latency)")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.3)

ax2.set_ylabel("queue reqs (sum)")
ax2.set_title("SGLang queued requests")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.3)

ax3.set_ylabel("KV util (avg/server)")
ax3.set_title("SGLang KV cache utilization (token_usage)")
ax3.axhline(1.0, color="gray", ls=":", alpha=0.6)
ax3.set_ylim(0, 1.02)
ax3.set_xlabel("time since first sample (s)")
ax3.legend(loc="lower center", fontsize=9)
ax3.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out, dpi=120)
print("saved:", out)
