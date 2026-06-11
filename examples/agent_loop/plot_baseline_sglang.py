import glob
import json
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

metrics_file = sys.argv[1]
out_png = sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else "SGLang metrics"


def base(key):
    return key.split("{", 1)[0]


# bin by 1s. Within a bin, each of the 8 DP engines is scraped several times, so
# average per (bin,target) first, THEN sum across targets to get the cluster total.
# acc[bin][metric][target] = [values...]
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
        vals = {}
        for k, v in s.items():
            vals[base(k)] = v
        b = int(mt)
        for m in ("sglang:num_running_reqs", "sglang:num_queue_reqs",
                  "sglang:num_used_tokens", "sglang:max_total_num_tokens"):
            if m in vals:
                acc[b][m][tgt].append(vals[m])


def bin_total(b, m):
    """mean per target, then sum across targets."""
    per_t = acc[b].get(m, {})
    return sum(np.mean(v) for v in per_t.values() if v)


run_bins = {b: bin_total(b, "sglang:num_running_reqs") for b in acc}
q_bins = {b: bin_total(b, "sglang:num_queue_reqs") for b in acc}
used_bins = {b: bin_total(b, "sglang:num_used_tokens") for b in acc}
max_bins = {b: bin_total(b, "sglang:max_total_num_tokens") for b in acc}

bins = sorted(acc)
t_start = min(bins)
t = np.array([b - t_start for b in bins], dtype=float)
running = np.array([run_bins.get(b, 0.0) for b in bins])
queue = np.array([q_bins.get(b, 0.0) for b in bins])
kv = np.array([(used_bins.get(b, 0.0) / max_bins[b] * 100.0) if max_bins.get(b) else 0.0 for b in bins])

fig, ax1 = plt.subplots(figsize=(13, 6))
ax1.set_xlabel("time (s)")
ax1.set_ylabel("# requests (summed over 8 DP engines)")
l1, = ax1.plot(t, running, color="tab:blue", lw=1.6, label="running reqs")
l2, = ax1.plot(t, queue, color="tab:red", lw=1.6, label="queue reqs")
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.set_ylabel("KV cache usage (%)")
l3, = ax2.plot(t, kv, color="tab:green", lw=1.6, alpha=0.8, label="KV usage %")
ax2.set_ylim(0, 100)

lines = [l1, l2, l3]
ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper right")
ax1.set_title(title)
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(out_png, dpi=110)
print("wrote", out_png)
print(f"running: mean={running.mean():.1f} max={running.max():.0f}")
print(f"queue:   mean={queue.mean():.1f} max={queue.max():.0f}")
print(f"KV%:     mean={kv.mean():.1f} max={kv.max():.1f}")
