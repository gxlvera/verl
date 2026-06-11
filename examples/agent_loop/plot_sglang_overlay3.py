import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# args: out_png  title  then triples of: label metrics_file color
out_png = sys.argv[1]
title = sys.argv[2]
rest = sys.argv[3:]
runs = [(rest[i], rest[i + 1], rest[i + 2]) for i in range(0, len(rest), 3)]


def base(key):
    return key.split("{", 1)[0]


def load(metrics_file):
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(metrics_file) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("series") or {}
            if not s:
                continue
            mt = d.get("monotonic_s")
            if mt is None:
                continue
            tgt = d.get("target", "?")
            vals = {base(k): v for k, v in s.items()}
            b = int(mt)
            for m in ("sglang:num_running_reqs", "sglang:num_queue_reqs",
                      "sglang:num_used_tokens", "sglang:max_total_num_tokens"):
                if m in vals:
                    acc[b][m][tgt].append(vals[m])

    def tot(b, m):
        return sum(np.mean(v) for v in acc[b].get(m, {}).values() if v)

    bins = sorted(acc)
    t0 = min(bins)
    t = np.array([b - t0 for b in bins], dtype=float)
    running = np.array([tot(b, "sglang:num_running_reqs") for b in bins])
    queue = np.array([tot(b, "sglang:num_queue_reqs") for b in bins])
    used = np.array([tot(b, "sglang:num_used_tokens") for b in bins])
    mx = np.array([tot(b, "sglang:max_total_num_tokens") for b in bins])
    kv = np.where(mx > 0, used / np.where(mx == 0, 1, mx) * 100.0, 0.0)
    return t, running, queue, kv


fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for label, f, color in runs:
    t, running, queue, kv = load(f)
    axes[0].plot(t, running, color=color, lw=1.5, label=label)
    axes[1].plot(t, queue, color=color, lw=1.5, label=label)
    axes[2].plot(t, kv, color=color, lw=1.5, label=label)

axes[0].set_ylabel("running reqs\n(sum 8 engines)")
axes[0].set_title(title)
axes[1].set_ylabel("queue reqs\n(sum 8 engines)")
axes[2].set_ylabel("KV cache usage (%)")
axes[2].set_ylim(0, 100)
axes[2].set_xlabel("time (s)")
for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
axes[2].set_ylim(0, 100)
fig.tight_layout()
fig.savefig(out_png, dpi=110)
print("wrote", out_png)
