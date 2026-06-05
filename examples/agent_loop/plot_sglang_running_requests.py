#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_current_targets(rows: list[dict]) -> set[str]:
    targets = []
    for row in rows:
        if row.get("event") in {"targets_discovered", "targets_refreshed"} and row.get("targets"):
            targets = row["targets"]
    return set(targets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot summed SGLang running requests from sampler JSONL.")
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

    by_t = defaultdict(lambda: {"running": 0.0, "queue": 0.0, "throughput": 0.0, "seq_lens": 0.0, "servers": 0})
    for row in rows:
        if row.get("target") not in current_targets or "summary" not in row:
            continue
        t = round(float(row["monotonic_s"]), 1)
        summary = row["summary"]
        rec = by_t[t]
        rec["running"] += summary.get("sglang:num_running_reqs", {}).get("last", 0.0)
        rec["queue"] += summary.get("sglang:num_queue_reqs", {}).get("last", 0.0)
        rec["throughput"] += summary.get("sglang:gen_throughput", {}).get("last", 0.0)
        rec["seq_lens"] += summary.get("sglang:decode_sum_seq_lens", {}).get("last", 0.0)
        rec["servers"] += 1

    all_points = [
        (t, values["running"], values["queue"], values["throughput"], values["seq_lens"], values["servers"])
        for t, values in sorted(by_t.items())
    ]
    if not all_points:
        raise SystemExit("No metric samples found for current targets.")

    active_indices = [
        i
        for i, point in enumerate(all_points)
        if point[1] > 0 or point[3] > 0 or point[4] > 0
    ]
    if not active_indices:
        active_start = all_points[0][0] if args.active_start is None else args.active_start
        active_end = all_points[-1][0] if args.active_end is None else args.active_end
    else:
        active_start = all_points[active_indices[0]][0] if args.active_start is None else args.active_start
        active_end = all_points[active_indices[-1]][0] if args.active_end is None else args.active_end

    points = [point for point in all_points if active_start <= point[0] <= active_end]
    if not points:
        raise SystemExit("No points remain after active-window filtering.")

    output_png = Path(args.output_png) if args.output_png else jsonl_path.with_name(jsonl_path.stem + "_running_reqs_active.png")
    output_csv = Path(args.output_csv) if args.output_csv else jsonl_path.with_name(jsonl_path.stem + "_running_reqs_active.csv")

    with output_csv.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "monotonic_s",
                "running_reqs_sum",
                "queue_reqs_sum",
                "gen_throughput_sum",
                "decode_sum_seq_lens_sum",
                "num_servers_sampled",
            ]
        )
        writer.writerows(points)

    xs = [point[0] for point in points]
    running = [point[1] for point in points]
    queue = [point[2] for point in points]

    zero_intervals = []
    start = None
    prev = None
    for point in points:
        if point[1] == 0:
            if start is None:
                start = point[0]
        elif start is not None:
            zero_intervals.append((start, prev))
            start = None
        prev = point[0]
    if start is not None:
        zero_intervals.append((start, points[-1][0]))

    plt.figure(figsize=(14, 5.5))
    plt.plot(xs, running, linewidth=1.5, color="#2563eb", label="running requests, sum over SGLang servers")
    plt.plot(xs, queue, linewidth=1.0, color="#f97316", alpha=0.8, label="queue requests, sum")
    for i, (gap_start, gap_end) in enumerate(zero_intervals):
        duration = gap_end - gap_start + 0.1
        if duration >= 0.5:
            plt.axvspan(
                gap_start,
                gap_end,
                color="#ef4444",
                alpha=0.18,
                label="running=0 gap >=0.5s" if i == 0 else None,
            )
    plt.title(args.title or f"SGLang Running Requests: {jsonl_path.name}")
    plt.xlabel("sampler monotonic time (s)")
    plt.ylabel("request count")
    plt.xlim(active_start, active_end)
    plt.ylim(bottom=0)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)

    print(f"jsonl={jsonl_path}")
    print(f"targets={sorted(current_targets)}")
    print(f"active_window={active_start:.1f},{active_end:.1f}")
    print(f"points={len(points)}")
    print(f"running_max={max(running)}")
    print(f"running_zero_points={sum(value == 0 for value in running)}")
    print(f"png={output_png}")
    print(f"csv={output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
