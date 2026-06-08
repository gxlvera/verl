#!/usr/bin/env python3
"""Plot the distribution of SGLang waiting-queue time for a given assistant turn.

Reads the per-turn queue-time JSONL produced by ToolAgentLoop when
AGENT_LOOP_QUEUE_TIME_JSONL is set (one file per worker pid). Each record:
    {"request_id", "turn", "queue_time_s", "prompt_len", "response_tokens", ...}

By default it focuses on turn 2 (the first tool-call return) and reports, over
all requests of that turn:
    - how many started prefill with ~no queueing (queue_time <= --zero-threshold)
    - how many queued, and the wait-time distribution (mean/median/p90/p99/max)
and saves a histogram of the non-trivial waits.

Usage:
    python plot_turn_queue_time.py --glob '/path/prefix*.jsonl' --turn 2 \
        --output-png turn2_queue.png
"""
import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def load(glob_pat: str, turn: int):
    rows = []
    for f in glob.glob(glob_pat):
        for line in Path(f).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("turn") == turn and r.get("queue_time_s") is not None:
                rows.append(r)
    return rows


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for queue-time JSONL files")
    ap.add_argument("--turn", type=int, default=2, help="assistant turn to analyze (2 = first tool-call return)")
    ap.add_argument("--zero-threshold", type=float, default=0.005,
                    help="queue_time_s <= this counts as 'no queueing' (seconds)")
    ap.add_argument("--output-png", default="turn_queue_time.png")
    ap.add_argument("--unit", choices=["s", "ms"], default="ms")
    args = ap.parse_args()

    rows = load(args.glob, args.turn)
    if not rows:
        raise SystemExit(f"No turn-{args.turn} records with queue_time found for glob {args.glob!r}")

    qts = [r["queue_time_s"] for r in rows]
    qts_sorted = sorted(qts)
    n = len(qts)
    no_queue = [q for q in qts if q <= args.zero_threshold]
    queued = [q for q in qts if q > args.zero_threshold]
    scale = 1000.0 if args.unit == "ms" else 1.0
    u = args.unit

    print(f"turn={args.turn}  total_requests={n}")
    print(f"no-queue (<= {args.zero_threshold*scale:.1f}{u}): {len(no_queue)} ({100*len(no_queue)/n:.1f}%)")
    print(f"queued                : {len(queued)} ({100*len(queued)/n:.1f}%)")
    if queued:
        qs = sorted(queued)
        print(f"queued wait {u}: mean={statistics.mean(qs)*scale:.1f} median={statistics.median(qs)*scale:.1f} "
              f"p90={pct(qs,90)*scale:.1f} p99={pct(qs,99)*scale:.1f} max={max(qs)*scale:.1f}")
    print(f"all-requests wait {u}: mean={statistics.mean(qts)*scale:.1f} median={statistics.median(qts_sorted)*scale:.1f} "
          f"p90={pct(qts_sorted,90)*scale:.1f} p99={pct(qts_sorted,99)*scale:.1f} max={max(qts)*scale:.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    # Left: queued vs not bar
    ax1.bar(["no queue\n(<= %.0f%s)" % (args.zero_threshold * scale, u), "queued"],
            [len(no_queue), len(queued)], color=["#16a34a", "#dc2626"])
    for i, v in enumerate([len(no_queue), len(queued)]):
        ax1.text(i, v, f"{v}\n({100*v/n:.0f}%)", ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel("request count")
    ax1.set_title(f"turn {args.turn}: queued vs not (N={n})")
    ax1.set_ylim(0, max(len(no_queue), len(queued)) * 1.18)

    # Right: histogram of queued wait times
    if queued:
        ax2.hist([q * scale for q in queued], bins=40, color="#dc2626", alpha=0.85)
        ax2.set_xlabel(f"waiting-queue time ({u})")
        ax2.set_ylabel("request count")
        ax2.set_title(f"turn {args.turn}: wait-time distribution among queued ({len(queued)})")
        ax2.axvline(statistics.median(queued) * scale, color="black", ls="--", lw=1,
                    label=f"median {statistics.median(queued)*scale:.0f}{u}")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "no queued requests", ha="center")

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=170)
    print(f"png={args.output_png}")


if __name__ == "__main__":
    raise SystemExit(main())
