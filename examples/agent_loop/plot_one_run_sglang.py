#!/usr/bin/env python3
"""Single-run SGLang figure: running reqs, queue reqs, and KV usage over time."""
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
TAG = sys.argv[1] if len(sys.argv) > 1 else "think_8192_500pt_shadow_20260608_132248"
TITLE = sys.argv[2] if len(sys.argv) > 2 else "full shadow (every turn), bs256"


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


fp = glob.glob(f"{LOG}/baseline_*{TAG}_bs256*_sglang_metrics.jsonl")[0]
run = defaultdict(list)
que = defaultdict(list)
tu = defaultdict(list)
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
ax1.plot(T, run_s, color="tab:blue", label="running reqs")
ax1.plot(T, que_s, color="tab:red", label="queue reqs")
ax1.set_ylabel("requests (sum over servers)")
ax1.set_title(f"{TITLE}: SGLang running / queue / KV usage")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

ax2.plot(T, tu_a, color="tab:orange", label="token_usage (KV util, avg/server)")
ax2.axhline(1.0, color="gray", ls=":", alpha=0.6)
ax2.set_ylabel("KV util")
ax2.set_xlabel("time since first sample (s)")
ax2.set_ylim(0, 1.02)
ax2.legend(loc="lower center")
ax2.grid(alpha=0.3)

fig.tight_layout()
out = f"{LOG}/sglang_onerun_{TAG}_bs256.png"
fig.savefig(out, dpi=120)
print("saved:", out)
print(f"peak running={max(run_s):.0f}  peak queue={max(que_s):.0f}  peak KV={max(tu_a):.3f}  window={T[-1]:.1f}s")
