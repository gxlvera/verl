#!/usr/bin/env python3
"""Compare a 'shadow non-thinking' bs256 run against the no-shadow baseline.

Shadow run: every assistant turn fires an extra parallel enable_thinking=False
request whose output is discarded (pure load). We compare main-trajectory
throughput (token/s, traj/s), peak SGLang running/queue, and turn-2 queueing.
"""
import csv
import glob
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/tiger/verl/logs"
BS = 256

RUNS = {
    "baseline (no shadow)": "think_8192_500pt_4tool_20260608_111923",
    "full shadow (every turn)": "think_8192_500pt_shadow_20260608_132248",
    "tail shadow (turn>=3 & tail)": "think_8192_500pt_tailshadow_20260608_140038",
}


def load_running_csv(tag):
    path = glob.glob(f"{LOG}/baseline_*{tag}_bs256*_sglang_metrics_running_reqs_active.csv")
    path = [p for p in path if "plot_active" not in p][0]
    t, run, que = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["monotonic_s"]))
            run.append(float(row["running_reqs_sum"]))
            que.append(float(row["queue_reqs_sum"]))
    return t, run, que


def active_window(t, run, thr=1.0):
    idx = [i for i, r in enumerate(run) if r >= thr]
    if not idx:
        return None
    return t[idx[-1]] - t[idx[0]]


def perf_throughput(tag):
    import re

    cand = [
        p
        for p in glob.glob(f"{LOG}/baseline_*{tag}_bs256*.log")
        if not any(x in p for x in ("plot_active", "sampler", "metrics"))
    ]
    L = min(cand, key=len)
    val = None
    pat = re.compile(r"perf/throughput['\"]?\s*[:=]\s*([0-9.]+)")
    with open(L, errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                val = float(m.group(1))
    return val


def turn2_queue(tag):
    files = glob.glob(f"{LOG}/qtime_{tag}_bs256*.jsonl")
    waits = []
    for fp in files:
        with open(fp, errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("turn") == 2:
                    waits.append(float(d.get("queue_time_s", 0.0)))
    if not waits:
        return None
    queued = [w for w in waits if w > 1e-3]
    return {
        "n": len(waits),
        "pct_queued": 100.0 * len(queued) / len(waits),
        "mean_all": statistics.mean(waits),
        "median_all": statistics.median(waits),
        "mean_queued": statistics.mean(queued) if queued else 0.0,
        "p90_queued": (sorted(queued)[int(0.9 * len(queued))] if queued else 0.0),
    }


fig, axes = plt.subplots(1, len(RUNS), figsize=(7 * len(RUNS), 5), sharey=True)
summary = {}

for ax, (label, tag) in zip(axes, RUNS.items()):
    t, run, que = load_running_csv(tag)
    t0 = t[0]
    tt = [x - t0 for x in t]
    ax.plot(tt, run, color="tab:blue", label="running reqs")
    ax.plot(tt, que, color="tab:red", label="queue reqs")
    ax.set_title(label)
    ax.set_xlabel("time since first sample (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    win = active_window(t, run)
    summary[label] = {
        "tag": tag,
        "tok_per_s": perf_throughput(tag),
        "active_window_s": win,
        "traj_per_s": (BS / win) if win else None,
        "peak_running": max(run),
        "peak_queue": max(que),
        "turn2": turn2_queue(tag),
    }

axes[0].set_ylabel("requests")
fig.suptitle("bs256, max_len=8192, per_turn=500, thinking ON: baseline vs +shadow non-thinking request")
fig.tight_layout()
out = f"{LOG}/shadow_vs_baseline_bs256_running_queue.png"
fig.savefig(out, dpi=120)
print("saved plot:", out)
print()

hdr = f"{'metric':<22}" + "".join(f"{lbl:>30}" for lbl in RUNS)
print(hdr)
print("-" * len(hdr))


def row(name, fmt, key, sub=None):
    cells = ""
    for lbl in RUNS:
        s = summary[lbl]
        v = s[key] if sub is None else (s["turn2"][sub] if s["turn2"] else None)
        cells += f"{(fmt.format(v) if v is not None else 'n/a'):>30}"
    print(f"{name:<22}{cells}")


row("tokens/s (main)", "{:.0f}", "tok_per_s")
row("rollout window (s)", "{:.1f}", "active_window_s")
row("traj/s", "{:.2f}", "traj_per_s")
row("peak running reqs", "{:.0f}", "peak_running")
row("peak queue reqs", "{:.0f}", "peak_queue")
row("turn2 % queued", "{:.1f}", None, "pct_queued")
row("turn2 mean wait all", "{:.3f}", None, "mean_all")
row("turn2 mean wait queued", "{:.3f}", None, "mean_queued")
row("turn2 p90 queued", "{:.3f}", None, "p90_queued")
