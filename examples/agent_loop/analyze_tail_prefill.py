#!/usr/bin/env python3
"""Is the bs256 long tail multi-turn (prefill still happening) or pure decode?

Plots, on one timeline: running reqs, queue reqs, prefill-queue reqs,
gen_throughput (decode tok/s), decode_sum_seq_lens, token_usage. Also dumps
turn distribution from the qtime jsonls.
"""
import glob
import json
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/tiger/verl/logs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "think_8192_500pt_4tool_20260608_111923"


def strip(k):
    return re.sub(r"\{.*?\}", "", k)


def load_series():
    fp = glob.glob(f"{LOG}/baseline_*{TAG}_bs256*_sglang_metrics.jsonl")[0]
    rows = []
    with open(fp, errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "series" not in d:
                continue
            agg = {}
            for k, v in d["series"].items():
                agg[strip(k)] = agg.get(strip(k), 0.0) + (v or 0.0)
            agg["t"] = d["monotonic_s"]
            rows.append(agg)
    rows.sort(key=lambda r: r["t"])
    return rows


rows = load_series()
# active window: running >= 1
run_idx = [i for i, r in enumerate(rows) if r.get("sglang:num_running_reqs", 0) >= 1]
i0, i1 = run_idx[0], run_idx[-1]
t0 = rows[i0]["t"]
rows = rows[i0 : i1 + 1]
T = [r["t"] - t0 for r in rows]


def col(key):
    return [r.get(key, 0.0) for r in rows]


running = col("sglang:num_running_reqs")
queue = col("sglang:num_queue_reqs")
prefill_q = [a + b for a, b in zip(col("sglang:num_prefill_inflight_queue_reqs"), col("sglang:num_prefill_prealloc_queue_reqs"))]
gen_thr = col("sglang:gen_throughput")
dssl = col("sglang:decode_sum_seq_lens")
tok_use = col("sglang:token_usage")

window = T[-1]
tail_start = 0.6 * window  # define "tail" as last 40% of wall time
tail = [i for i, t in enumerate(T) if t >= tail_start]


def avg(xs, idx):
    vals = [xs[i] for i in idx if xs[i] is not None]
    return statistics.mean(vals) if vals else 0.0


def mx(xs, idx):
    vals = [xs[i] for i in idx]
    return max(vals) if vals else 0.0


fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
ax[0].plot(T, running, color="tab:blue", label="running reqs")
ax[0].plot(T, queue, color="tab:red", label="queue reqs")
ax[0].plot(T, prefill_q, color="tab:green", label="prefill-queue reqs")
ax[0].axvline(tail_start, color="gray", ls="--", alpha=0.7)
ax[0].set_ylabel("reqs")
ax[0].legend(loc="upper right")
ax[0].set_title(f"{TAG} bs256 -- tail (last 40%) marked by dashed line")

ax[1].plot(T, gen_thr, color="tab:purple", label="gen_throughput (decode tok/s)")
ax[1].axvline(tail_start, color="gray", ls="--", alpha=0.7)
ax[1].set_ylabel("decode tok/s")
ax[1].legend(loc="upper right")

ax[2].plot(T, tok_use, color="tab:orange", label="token_usage (KV util)")
ax[2].axvline(tail_start, color="gray", ls="--", alpha=0.7)
ax[2].set_ylabel("KV util")
ax[2].set_xlabel("time since first sample (s)")
ax[2].legend(loc="upper right")
ax2b = ax[2].twinx()
ax2b.plot(T, dssl, color="tab:gray", alpha=0.6, label="decode_sum_seq_lens")
ax2b.set_ylabel("decode_sum_seq_lens")
ax2b.legend(loc="upper left")

fig.tight_layout()
out = f"{LOG}/tail_prefill_{TAG}_bs256.png"
fig.savefig(out, dpi=120)
print("saved:", out)

print(f"\n== window {window:.1f}s, tail = last 40% ([{tail_start:.0f}s .. {window:.0f}s]) ==")
print(f"prefill-queue reqs : mean(all)={avg(prefill_q, range(len(T))):.2f}  max(all)={mx(prefill_q, range(len(T))):.0f}  | tail mean={avg(prefill_q, tail):.2f} tail max={mx(prefill_q, tail):.0f}")
print(f"queue reqs         : tail mean={avg(queue, tail):.1f}  tail max={mx(queue, tail):.0f}")
print(f"token_usage        : peak={max(tok_use):.2f}  tail mean={avg(tok_use, tail):.2f}")
print(f"gen_throughput     : peak={max(gen_thr):.0f}  tail mean={avg(gen_thr, tail):.0f} tok/s")
print(f"running reqs       : peak={max(running):.0f}  tail mean={avg(running, tail):.0f}")

# turn distribution from qtime
qfiles = glob.glob(f"{LOG}/qtime_{TAG}_bs256*.jsonl")
per_req = {}
turn_resp = {}
for fp in qfiles:
    with open(fp, errors="ignore") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            rid = d.get("request_id")
            tn = d.get("turn")
            per_req[rid] = max(per_req.get(rid, 0), tn)
            turn_resp.setdefault(tn, []).append(d.get("response_tokens", 0))

if per_req:
    turns = list(per_req.values())
    from collections import Counter

    dist = Counter(turns)
    print(f"\n== turn distribution over {len(turns)} requests ==")
    print(f"avg turns/req = {statistics.mean(turns):.2f}")
    for k in sorted(dist):
        print(f"  reached turn {k}: {dist[k]:4d} reqs ({100*dist[k]/len(turns):4.1f}%)")
    print("\n== response tokens per turn (decode work) ==")
    for k in sorted(turn_resp):
        v = turn_resp[k]
        print(f"  turn {k}: n={len(v):4d}  mean={statistics.mean(v):6.1f}  p90={sorted(v)[int(0.9*len(v))-1]:6.0f}  max={max(v):6.0f}")
