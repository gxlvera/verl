#!/usr/bin/env python3
"""Sweep B (max_len=4096, cap=2) analysis: throughput/cache table + per-bs SGLang plots.

Same methodology as Sweep A:
  - traj/s   = (bs * rollout_n) / timing_s/gen
  - token/s  = perf/total_num_tokens / timing_s/gen   (full machine, 8 GPUs)
  - cache_hit (median) = median over active window of mean(sglang:cache_hit_rate)
  - KV peak  = max over active window of mean(sglang:token_usage)
  - evicted  = final cumulative sum(sglang:evicted_tokens_total) over servers

Per-bs plot: running req + queue req (left axis) and KV token_usage (right axis),
three metrics on one figure, one figure per bs.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLLOUT_N = 8
LOGDIR = Path("/home/tiger/verl/logs")
TAG = "rolloutonly_4096_cap2_extra_20260607_174249"
BS_LIST = [160, 192, 224, 256, 384, 512, 1024, 2048]

RUNNING = "sglang:num_running_reqs"
QUEUE = "sglang:num_queue_reqs"
USED = "sglang:num_used_tokens"
USAGE = "sglang:token_usage"
CACHE_HIT = "sglang:cache_hit_rate"
EVICTED = "sglang:evicted_tokens_total"


def _last(summary, key):
    v = summary.get(key)
    if isinstance(v, dict):
        return float(v.get("last", 0.0) or 0.0)
    return 0.0


def load_current_targets(rows):
    targets = []
    for row in rows:
        if row.get("event") in {"targets_discovered", "targets_refreshed"} and row.get("targets"):
            targets = row["targets"]
    return set(targets)


def step1_metrics(log_path):
    gen = total = None
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            if "step:1 " not in line:
                continue
            for tok in line.split():
                if tok.startswith("timing_s/gen:"):
                    gen = float(tok.split(":", 1)[1])
                elif tok.startswith("perf/total_num_tokens:"):
                    total = float(tok.split(":", 1)[1])
    return gen, total


def analyze_run(bs):
    log_path = LOGDIR / f"baseline_hotpotqa_qwen3_8b_sleep0_toolcalls1_bs{bs}_rollout8_step1_{TAG}_bs{bs}.log"
    jsonl_path = LOGDIR / f"baseline_hotpotqa_qwen3_8b_sleep0_toolcalls1_bs{bs}_rollout8_step1_{TAG}_bs{bs}_sglang_metrics.jsonl"

    gen, total = step1_metrics(log_path)
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    targets = load_current_targets(rows)

    by_t = defaultdict(lambda: {"running": 0.0, "queue": 0.0, "used": 0.0,
                                "usage_sum": 0.0, "hit_sum": 0.0, "evicted": 0.0, "servers": 0})
    for row in rows:
        if row.get("target") not in targets or "summary" not in row:
            continue
        s = row["summary"]
        t = round(float(row["monotonic_s"]), 1)
        rec = by_t[t]
        rec["running"] += _last(s, RUNNING)
        rec["queue"] += _last(s, QUEUE)
        rec["used"] += _last(s, USED)
        rec["usage_sum"] += _last(s, USAGE)
        rec["hit_sum"] += _last(s, CACHE_HIT)
        rec["evicted"] += _last(s, EVICTED)
        rec["servers"] += 1

    pts = []
    for t, v in sorted(by_t.items()):
        ns = v["servers"] or 1
        pts.append({
            "t": t, "running": v["running"], "queue": v["queue"], "used": v["used"],
            "usage": v["usage_sum"] / ns, "hit": v["hit_sum"] / ns, "evicted": v["evicted"],
        })

    # active window: KV in use
    active = [p for p in pts if p["used"] > 0]
    if not active:
        active = [p for p in pts if p["running"] > 0]
    t0 = active[0]["t"]
    t1 = active[-1]["t"]
    win = [p for p in pts if t0 <= p["t"] <= t1]

    hits = [p["hit"] for p in win if p["hit"] > 0]
    cache_hit_med = statistics.median(hits) if hits else 0.0
    kv_peak = max((p["usage"] for p in win), default=0.0)
    evicted_final = max((p["evicted"] for p in win), default=0.0)

    trajectories = bs * ROLLOUT_N
    traj_s = trajectories / gen if gen else 0.0
    token_s = total / gen if gen else 0.0

    # ---- per-bs plot: running + queue (left) + kv usage (right) ----
    xs = [p["t"] - t0 for p in win]
    running = [p["running"] for p in win]
    queue = [p["queue"] for p in win]
    usage = [p["usage"] for p in win]

    fig, axl = plt.subplots(figsize=(14, 5.5))
    axl.plot(xs, running, linewidth=1.6, color="#2563eb", label="running requests (sum over servers)")
    axl.plot(xs, queue, linewidth=1.3, color="#f59e0b", label="queue requests (sum over servers)")
    axl.set_xlabel("time since active start (s)")
    axl.set_ylabel("request count")
    axl.set_xlim(0, max(xs) if xs else 1)
    axl.set_ylim(bottom=0)
    axl.grid(True, axis="y", alpha=0.25)

    axr = axl.twinx()
    axr.plot(xs, usage, linewidth=1.6, color="#16a34a", alpha=0.85, label="KV usage (token_usage, mean)")
    axr.set_ylabel("KV usage (fraction 0..1)")
    axr.set_ylim(0, 1)

    ll, la = axl.get_legend_handles_labels()
    rl, ra = axr.get_legend_handles_labels()
    axl.legend(ll + rl, la + ra, loc="upper right")
    plt.title(f"Sweep B  bs{bs}  (max_len=4096, cap=2): running / queue / KV usage")
    fig.tight_layout()
    out_png = LOGDIR / f"B_sglang_bs{bs}_4096cap2.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {
        "bs": bs, "trajectories": trajectories, "gen": gen, "total": total,
        "traj_s": traj_s, "token_s": token_s, "cache_hit_med": cache_hit_med,
        "kv_peak": kv_peak, "evicted": evicted_final, "running_max": max(running, default=0),
        "queue_max": max(queue, default=0), "png": str(out_png),
    }


def main():
    results = []
    for bs in BS_LIST:
        r = analyze_run(bs)
        results.append(r)
        print(f"bs={bs:5d}  gen={r['gen']:.1f}s  traj/s={r['traj_s']:.1f}  "
              f"token/s={r['token_s']:.0f}  hit_med={r['cache_hit_med']:.2f}  "
              f"KVpeak={r['kv_peak']*100:.0f}%  evicted={r['evicted']/1e6:.1f}M  "
              f"run_max={r['running_max']:.0f} q_max={r['queue_max']:.0f}")

    print("\n==== Sweep B throughput + cache table (markdown) ====\n")
    print("| bs | trajectories | traj/s | token/s | cache_hit (median) | KV peak | cumulative evicted |")
    print("|----|----|----|----|----|----|----|")
    for r in results:
        print(f"| {r['bs']} | {r['trajectories']} | {r['traj_s']:.1f} | {r['token_s']:,.0f} | "
              f"{r['cache_hit_med']:.2f} | {r['kv_peak']*100:.0f}% | {r['evicted']/1e6:.1f}M |")


if __name__ == "__main__":
    main()
