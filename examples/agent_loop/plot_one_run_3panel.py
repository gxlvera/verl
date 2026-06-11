#!/usr/bin/env python3
"""Single-run 3-panel SGLang figure: running / queue / KV usage.
Usage: python plot_one_run_3panel.py <metrics.jsonl> <title> <out.png>
"""
import json
import re
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fp, title, out = sys.argv[1], sys.argv[2], sys.argv[3]


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


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
sl = slice(i0, i1 + 1)
ts, run_s, que_s, tu_a = ts[sl], run_s[sl], que_s[sl], tu_a[sl]
t0 = ts[0]
T = [t - t0 for t in ts]

fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
a1.plot(T, run_s, color="tab:blue")
a1.set_ylabel("running reqs (sum)")
a1.set_title(f"{title}: running requests  (peak={max(run_s):.0f}, makespan={T[-1]:.0f}s)")
a1.grid(alpha=0.3)
a2.plot(T, que_s, color="tab:red")
a2.set_ylabel("queue reqs (sum)")
a2.set_title(f"queued requests  (peak={max(que_s):.0f})")
a2.grid(alpha=0.3)
a3.plot(T, tu_a, color="tab:orange")
a3.axhline(1.0, color="gray", ls=":", alpha=0.6)
a3.set_ylim(0, 1.02)
a3.set_ylabel("KV util (avg/server)")
a3.set_title(f"KV cache utilization  (peak={max(tu_a):.3f})")
a3.set_xlabel("time since first sample (s)")
a3.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=120)
print("saved:", out)
print(f"peak running={max(run_s):.0f} peak queue={max(que_s):.0f} peak KV={max(tu_a):.3f} makespan={T[-1]:.1f}s")
