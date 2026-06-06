#!/usr/bin/env python3
"""Plot SGLang KV-cache token usage over time from the sampler JSONL.

Reads a `*_sglang_metrics.jsonl` file produced by sample_sglang_metrics.py and
plots, aggregated across all SGLang servers (rollout replicas) per timestamp:

    sglang:num_used_tokens      summed over servers   (left axis, counts)
    sglang:max_total_num_tokens summed over servers   (left axis, ceiling line)
    sglang:token_usage          mean over servers      (right axis, 0..1)

token_usage is a per-server ratio, so summing it is meaningless; we report the
mean across servers and also an aggregate ratio = used_sum / max_total_sum,
which equals the mean when every server has the same capacity.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


USED = "sglang:num_used_tokens"
MAX_TOTAL = "sglang:max_total_num_tokens"
USAGE = "sglang:token_usage"


def load_current_targets(rows: list[dict]) -> set[str]:
    targets = []
    for row in rows:
        if row.get("event") in {"targets_discovered", "targets_refreshed"} and row.get("targets"):
            targets = row["targets"]
    return set(targets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot SGLang KV-cache token usage from sampler JSONL.")
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

    by_t = defaultdict(lambda: {"used": 0.0, "max_total": 0.0, "usage_sum": 0.0, "servers": 0})
    saw_token_metric = False
    for row in rows:
        if row.get("target") not in current_targets or "summary" not in row:
            continue
        summary = row["summary"]
        if USED in summary or USAGE in summary or MAX_TOTAL in summary:
            saw_token_metric = True
        t = round(float(row["monotonic_s"]), 1)
        rec = by_t[t]
        rec["used"] += summary.get(USED, {}).get("last", 0.0)
        rec["max_total"] += summary.get(MAX_TOTAL, {}).get("last", 0.0)
        rec["usage_sum"] += summary.get(USAGE, {}).get("last", 0.0)
        rec["servers"] += 1

    if not saw_token_metric:
        raise SystemExit(
            "No token metrics (num_used_tokens/token_usage/max_total_num_tokens) in this JSONL.\n"
            "This run predates the DEFAULT_PATTERNS update; re-run to capture them."
        )

    all_points = []
    for t, v in sorted(by_t.items()):
        servers = v["servers"] or 1
        usage_mean = v["usage_sum"] / servers
        usage_agg = v["used"] / v["max_total"] if v["max_total"] > 0 else 0.0
        all_points.append((t, v["used"], v["max_total"], usage_mean, usage_agg, v["servers"]))
    if not all_points:
        raise SystemExit("No metric samples found for current targets.")

    active_indices = [i for i, p in enumerate(all_points) if p[1] > 0]
    if not active_indices:
        active_start = all_points[0][0] if args.active_start is None else args.active_start
        active_end = all_points[-1][0] if args.active_end is None else args.active_end
    else:
        active_start = all_points[active_indices[0]][0] if args.active_start is None else args.active_start
        active_end = all_points[active_indices[-1]][0] if args.active_end is None else args.active_end

    points = [p for p in all_points if active_start <= p[0] <= active_end]
    if not points:
        raise SystemExit("No points remain after active-window filtering.")

    output_png = Path(args.output_png) if args.output_png else jsonl_path.with_name(jsonl_path.stem + "_token_usage_active.png")
    output_csv = Path(args.output_csv) if args.output_csv else jsonl_path.with_name(jsonl_path.stem + "_token_usage_active.csv")

    with output_csv.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "monotonic_s",
                "num_used_tokens_sum",
                "max_total_num_tokens_sum",
                "token_usage_mean",
                "token_usage_aggregate",
                "num_servers_sampled",
            ]
        )
        writer.writerows(points)

    xs = [p[0] for p in points]
    used = [p[1] for p in points]
    max_total = [p[2] for p in points]
    usage_mean = [p[3] for p in points]

    fig, ax_left = plt.subplots(figsize=(14, 5.5))
    ax_left.plot(xs, used, linewidth=1.5, color="#2563eb", label="num_used_tokens, sum over servers")
    ax_left.plot(xs, max_total, linewidth=1.0, color="#9ca3af", linestyle="--", label="max_total_num_tokens, sum (capacity)")
    ax_left.set_xlabel("sampler monotonic time (s)")
    ax_left.set_ylabel("KV tokens (summed over servers)")
    ax_left.set_xlim(active_start, active_end)
    ax_left.set_ylim(bottom=0)
    ax_left.grid(True, axis="y", alpha=0.25)

    ax_right = ax_left.twinx()
    ax_right.plot(xs, usage_mean, linewidth=1.5, color="#16a34a", alpha=0.85, label="token_usage, mean over servers")
    ax_right.set_ylabel("token_usage (fraction)")
    ax_right.set_ylim(0, 1)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="upper right")

    plt.title(args.title or f"SGLang KV-cache token usage: {jsonl_path.name}")
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)

    peak_idx = max(range(len(points)), key=lambda i: points[i][1])
    print(f"jsonl={jsonl_path}")
    print(f"targets={sorted(current_targets)}")
    print(f"active_window={active_start:.1f},{active_end:.1f}")
    print(f"points={len(points)}")
    print(f"num_used_tokens_sum_max={max(used):.0f} at t={points[peak_idx][0]:.1f}s")
    print(f"max_total_num_tokens_sum_typical={max(max_total):.0f}")
    print(f"token_usage_mean_max={max(usage_mean):.3f}")
    print(f"png={output_png}")
    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
