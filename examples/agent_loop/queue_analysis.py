#!/usr/bin/env python3
"""Per-turn SGLang queue-time analysis from qtime_*.jsonl.

Usage: python queue_analysis.py <qtime_tag_prefix> [queued_threshold_s]
Reads /home/tiger/verl/logs/<prefix>.*.jsonl
"""
import glob
import json
import sys
from collections import defaultdict

import numpy as np

LOG = "/home/tiger/verl/logs"
prefix = sys.argv[1]
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05  # >50ms counts as "queued"

files = glob.glob(f"{LOG}/{prefix}.*.jsonl")
by_turn = defaultdict(list)
allq = []
for fp in files:
    for line in open(fp, errors="ignore"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        q = d.get("queue_time_s")
        if q is None:
            continue
        by_turn[d.get("turn", -1)].append(q)
        allq.append(q)


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else 0.0


def summarize(name, arr):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    q = arr[arr > THR]  # queued subset only
    fq = len(q) / n * 100 if n else 0.0
    print(
        f"{name:>7s}  n={n:6d}  queued={len(q):6d} ({fq:5.1f}%)  | among QUEUED: "
        f"mean={ (q.mean() if len(q) else 0):7.3f}s  p50={pct(q,50):7.3f}  "
        f"p90={pct(q,90):7.3f}  p99={pct(q,99):7.3f}  max={ (q.max() if len(q) else 0):6.2f}s"
    )


print(f"=== {prefix}  (queued threshold = {THR*1000:.0f} ms) ===")
print(f"turns present: {sorted(t for t in by_turn if t>=0)}")
summarize("ALL", allq)
for t in sorted(by_turn):
    if t < 0:
        continue
    summarize(f"turn{t}", by_turn[t])
