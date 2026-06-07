#!/usr/bin/env python3
"""Overlay num_running_reqs + num_queue_reqs (summed over SGLang servers) for
several runs on one time axis, plus a panel of cumulative evicted tokens.

Each run's time axis is shifted so t=0 is the start of its active window (first
timestamp with num_used_tokens > 0), so runs of different durations line up at
their rollout start.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RUNNING = "sglang:num_running_reqs"
QUEUE = "sglang:num_queue_reqs"
USED = "sglang:num_used_tokens"
EVICTED = "sglang:evicted_tokens_total"


def load_targets(rows):
    targets = []
    for r in rows:
        if r.get("event") in {"targets_discovered", "targets_refreshed"} and r.get("targets"):
            targets = r["targets"]
    return set(targets)


def _last(summary, key):
    v = summary.get(key)
    if isinstance(v, dict):
        return float(v.get("last", 0.0) or 0.0)
    return 0.0


def series(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    targets = load_targets(rows)
    by_t = defaultdict(lambda: {"run": 0.0, "q": 0.0, "used": 0.0, "evict": 0.0})
    for r in rows:
        if r.get("target") not in targets or "summary" not in r:
            continue
        s = r["summary"]
        t = round(float(r["monotonic_s"]), 1)
        rec = by_t[t]
        rec["run"] += _last(s, RUNNING)
        rec["q"] += _last(s, QUEUE)
        rec["used"] += _last(s, USED)
        rec["evict"] += _last(s, EVICTED)
    pts = sorted(by_t.items())
    active = [t for t, v in pts if v["used"] > 0]
    if not active:
        active = [pts[0][0], pts[-1][0]]
    t0, t1 = active[0], active[-1]
    out = []
    for t, v in pts:
        if t0 <= t <= t1:
            out.append((t - t0, v["run"], v["q"], v["evict"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="LABEL=path.jsonl")
    ap.add_argument("--output-png", required=True)
    ap.add_argument("--title", default="SGLang running/queue overlay")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        runs.append((label, series(path)))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for (label, pts), c in zip(runs, colors):
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1] for p in pts], color=c, lw=1.6, label=f"{label} running")
        ax1.plot(xs, [p[2] for p in pts], color=c, lw=1.1, ls="--", alpha=0.8,
                 label=f"{label} queue")
    ax1.set_ylabel("request count (sum over servers)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", ncol=len(runs), fontsize=8)
    ax1.set_title(args.title)

    for (label, pts), c in zip(runs, colors):
        xs = [p[0] for p in pts]
        ax2.plot(xs, [p[3] / 1e6 for p in pts], color=c, lw=1.6, label=f"{label} evicted")
    ax2.set_ylabel("cumulative evicted tokens (millions)")
    ax2.set_xlabel("time since rollout start (s)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=170)
    print(f"png={args.output_png}")
    for label, pts in runs:
        rp = max((p[1] for p in pts), default=0)
        qp = max((p[2] for p in pts), default=0)
        ev = max((p[3] for p in pts), default=0)
        print(f"{label}: running_peak={rp:.0f} queue_peak={qp:.0f} evicted_final={ev:.0f}")


if __name__ == "__main__":
    raise SystemExit(main())
