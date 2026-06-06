#!/usr/bin/env python3
"""Plot SGLang preemption/retraction *proxy* signals over time.

SGLang does not (in this scrape) export a direct `num_retracted_reqs` counter,
so we infer when the scheduler is under KV pressure / likely retracting decode
batches by overlaying, aggregated across all SGLang servers per timestamp:

    left axis (counts):
        sglang:num_running_reqs   sum over servers
        sglang:num_queue_reqs     sum over servers
    right axis (0..1 fractions):
        sglang:token_usage                  mean over servers (KV occupancy)
        sglang:pending_prealloc_token_usage mean over servers (prealloc pressure)

Retraction signature: num_running_reqs drops sharply while token_usage stays
pinned near 1.0 (KV full) and pending_prealloc_token_usage spikes -- i.e. the
engine cannot grow the running batch and evicts/queues requests instead.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RUNNING = "sglang:num_running_reqs"
QUEUE = "sglang:num_queue_reqs"
USED = "sglang:num_used_tokens"
MAX_TOTAL = "sglang:max_total_num_tokens"
USAGE = "sglang:token_usage"
PREALLOC = "sglang:pending_prealloc_token_usage"


def load_current_targets(rows: list[dict]) -> set[str]:
    targets = []
    for row in rows:
        if row.get("event") in {"targets_discovered", "targets_refreshed"} and row.get("targets"):
            targets = row["targets"]
    return set(targets)


def _last(summary: dict, key: str) -> float:
    v = summary.get(key)
    if isinstance(v, dict):
        return float(v.get("last", 0.0) or 0.0)
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot SGLang preemption/retraction proxy signals from sampler JSONL.")
    parser.add_argument("--jsonl", required=True, help="SGLang metrics JSONL file")
    parser.add_argument("--output-png", default="", help="Output PNG path")
    parser.add_argument("--output-csv", default="", help="Output CSV path")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--active-start", type=float, default=None, help="Override active window start")
    parser.add_argument("--active-end", type=float, default=None, help="Override active window end")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    current_targets = load_current_targets(rows)
    if not current_targets:
        raise SystemExit("No SGLang targets found in JSONL events.")

    by_t = defaultdict(lambda: {"running": 0.0, "queue": 0.0, "used": 0.0,
                                "max_total": 0.0, "usage_sum": 0.0,
                                "prealloc_sum": 0.0, "servers": 0})
    for row in rows:
        if row.get("target") not in current_targets or "summary" not in row:
            continue
        summary = row["summary"]
        t = round(float(row["monotonic_s"]), 1)
        rec = by_t[t]
        rec["running"] += _last(summary, RUNNING)
        rec["queue"] += _last(summary, QUEUE)
        rec["used"] += _last(summary, USED)
        rec["max_total"] += _last(summary, MAX_TOTAL)
        rec["usage_sum"] += _last(summary, USAGE)
        rec["prealloc_sum"] += _last(summary, PREALLOC)
        rec["servers"] += 1

    all_points = []
    for t, v in sorted(by_t.items()):
        servers = v["servers"] or 1
        all_points.append((
            t, v["running"], v["queue"], v["used"], v["max_total"],
            v["usage_sum"] / servers, v["prealloc_sum"] / servers, v["servers"],
        ))
    if not all_points:
        raise SystemExit("No metric samples found for current targets.")

    # Active window = where KV is actually in use.
    active_indices = [i for i, p in enumerate(all_points) if p[3] > 0]
    if active_indices:
        active_start = all_points[active_indices[0]][0] if args.active_start is None else args.active_start
        active_end = all_points[active_indices[-1]][0] if args.active_end is None else args.active_end
    else:
        active_start = all_points[0][0] if args.active_start is None else args.active_start
        active_end = all_points[-1][0] if args.active_end is None else args.active_end

    points = [p for p in all_points if active_start <= p[0] <= active_end]
    if not points:
        raise SystemExit("No points remain after active-window filtering.")

    output_png = Path(args.output_png) if args.output_png else jsonl_path.with_name(jsonl_path.stem + "_preempt_signals.png")
    output_csv = Path(args.output_csv) if args.output_csv else jsonl_path.with_name(jsonl_path.stem + "_preempt_signals.csv")

    with output_csv.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "monotonic_s", "num_running_reqs_sum", "num_queue_reqs_sum",
            "num_used_tokens_sum", "max_total_num_tokens_sum",
            "token_usage_mean", "pending_prealloc_token_usage_mean", "num_servers_sampled",
        ])
        writer.writerows(points)

    xs = [p[0] for p in points]
    running = [p[1] for p in points]
    queue = [p[2] for p in points]
    usage_mean = [p[5] for p in points]
    prealloc_mean = [p[6] for p in points]

    fig, ax_left = plt.subplots(figsize=(14, 5.5))
    ax_left.plot(xs, running, linewidth=1.5, color="#2563eb", label="num_running_reqs, sum")
    ax_left.plot(xs, queue, linewidth=1.2, color="#f59e0b", label="num_queue_reqs, sum")
    ax_left.set_xlabel("sampler monotonic time (s)")
    ax_left.set_ylabel("request count")
    ax_left.set_xlim(active_start, active_end)
    ax_left.set_ylim(bottom=0)
    ax_left.grid(True, axis="y", alpha=0.25)

    ax_right = ax_left.twinx()
    ax_right.plot(xs, usage_mean, linewidth=1.5, color="#16a34a", alpha=0.85, label="token_usage, mean")
    ax_right.plot(xs, prealloc_mean, linewidth=1.5, color="#dc2626", alpha=0.85, label="pending_prealloc_token_usage, mean")
    ax_right.set_ylabel("fraction (0..1)")
    ax_right.set_ylim(0, max(1.0, max(prealloc_mean) * 1.1 if prealloc_mean else 1.0))

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")

    plt.title(args.title or f"SGLang preempt/retract proxy signals: {jsonl_path.name}")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)

    # Heuristic onset: first time token_usage >= 0.95 AND running has dropped
    # below half of its early peak (i.e. KV full while concurrency collapsing).
    run_peak = max(running) if running else 0.0
    onset_t = None
    for (t, r, q, used, mxt, um, pa, ns) in points:
        if um >= 0.95 and run_peak > 0 and r <= 0.5 * run_peak:
            onset_t = t
            break

    print(f"jsonl={jsonl_path}")
    print(f"active_window={active_start:.1f},{active_end:.1f}")
    print(f"points={len(points)}")
    print(f"running_peak={run_peak:.0f}")
    print(f"token_usage_mean_max={max(usage_mean):.3f}")
    print(f"pending_prealloc_token_usage_mean_max={max(prealloc_mean):.4f}")
    if onset_t is not None:
        print(f"pressure_onset_t={onset_t:.1f}s (token_usage>=0.95 and running<=50% of peak)")
    else:
        print("pressure_onset_t=none (no point met token_usage>=0.95 & running<=50% peak)")
    print(f"png={output_png}")
    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
