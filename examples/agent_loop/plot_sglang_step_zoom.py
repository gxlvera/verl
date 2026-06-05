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


def load_points(jsonl_path: Path) -> list[tuple[float, float, float, float, float, int]]:
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

    points = [
        (t, values["running"], values["queue"], values["throughput"], values["seq_lens"], values["servers"])
        for t, values in sorted(by_t.items())
    ]
    if not points:
        raise SystemExit("No metric samples found for current targets.")
    return points


def active_points(points: list[tuple[float, float, float, float, float, int]]) -> list[tuple[float, float, float, float, float, int]]:
    active_indices = [i for i, point in enumerate(points) if point[1] > 0 or point[3] > 0 or point[4] > 0]
    if not active_indices:
        return points
    return points[active_indices[0] : active_indices[-1] + 1]


def peak_segments(points: list[tuple[float, float, float, float, float, int]], threshold: float) -> list[tuple[int, int, float, float, float]]:
    segs = []
    start = None
    for i, point in enumerate(points):
        above = point[1] > threshold
        if above and start is None:
            start = i
        last = i == len(points) - 1
        if start is not None and ((not above) or last):
            end = i - 1 if not above else i
            max_running = max(p[1] for p in points[start : end + 1])
            segs.append((start, end, points[start][0], points[end][0], max_running))
            start = None
    return segs


def merge_nearby_segments(
    segments: list[tuple[int, int, float, float, float]], gap_s: float
) -> list[tuple[int, int, float, float, float]]:
    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg[2] - prev[3]
        if gap <= gap_s:
            merged[-1] = (prev[0], seg[1], prev[2], seg[3], max(prev[4], seg[4]))
        else:
            merged.append(seg)
    return merged


def write_csv(path: Path, points: list[tuple[float, float, float, float, float, int]], window_start: float) -> None:
    with path.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "monotonic_s",
                "relative_s",
                "running_reqs_sum",
                "queue_reqs_sum",
                "gen_throughput_sum",
                "decode_sum_seq_lens_sum",
                "num_servers_sampled",
            ]
        )
        for point in points:
            writer.writerow([point[0], round(point[0] - window_start, 1), point[1], point[2], point[3], point[4], point[5]])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot step2/step3 SGLang running requests at 0.1s resolution.")
    parser.add_argument("--jsonl", required=True, help="SGLang metrics JSONL file")
    parser.add_argument("--output-png", default="", help="Output PNG path")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--steps", default="2,3", help="Comma-separated 1-indexed main peak steps to plot")
    parser.add_argument("--main-peak-threshold", type=float, default=120.0, help="Minimum peak value to count as a main rollout step")
    parser.add_argument("--segment-threshold", type=float, default=64.0, help="Running-request threshold used to find peak segments")
    parser.add_argument("--merge-gap-s", type=float, default=6.0, help="Merge nearby peak fragments separated by this many seconds")
    parser.add_argument("--lead-s", type=float, default=2.0, help="Seconds before main peak to include")
    parser.add_argument("--tail-s", type=float, default=8.0, help="Seconds after main peak to include")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    output_png = Path(args.output_png) if args.output_png else jsonl_path.with_name(jsonl_path.stem + "_step2_3_zoom_0p1s.png")
    requested_steps = [int(value) for value in args.steps.split(",") if value.strip()]

    points = active_points(load_points(jsonl_path))
    all_segments = peak_segments(points, args.segment_threshold)
    merged_segments = merge_nearby_segments(all_segments, args.merge_gap_s)
    main_segments = [seg for seg in merged_segments if seg[4] >= args.main_peak_threshold]
    if max(requested_steps) > len(main_segments):
        raise SystemExit(
            f"Requested steps {requested_steps}, but only found {len(main_segments)} main peaks. "
            f"all_segments={[(round(s[2], 1), round(s[3], 1), s[4]) for s in all_segments]}"
        )

    windows = []
    min_t, max_t = points[0][0], points[-1][0]
    for step in requested_steps:
        seg = main_segments[step - 1]
        _, _, seg_start, seg_end, max_running = seg
        window_start = round(max(min_t, seg_start - args.lead_s), 1)
        window_end = round(min(max_t, seg_end + args.tail_s), 1)
        sub_points = [point for point in points if window_start <= point[0] <= window_end]
        if not sub_points:
            raise SystemExit(f"No points found for step{step} window {window_start}-{window_end}.")
        windows.append((step, window_start, window_end, max_running, sub_points))
        csv_path = output_png.with_name(output_png.stem + f"_step{step}.csv")
        write_csv(csv_path, sub_points, window_start)

    y_max = max(point[1] for _step, _window_start, _window_end, _max_running, sub_points in windows for point in sub_points)
    y_limit = max(135, y_max + 5)

    fig, axes = plt.subplots(len(windows), 1, figsize=(15, 4 * len(windows)), sharey=True)
    if len(windows) == 1:
        axes = [axes]
    for ax, (step, window_start, window_end, _max_running, sub_points) in zip(axes, windows):
        rel = [round(point[0] - window_start, 1) for point in sub_points]
        running = [point[1] for point in sub_points]
        queue = [point[2] for point in sub_points]
        ax.plot(rel, running, color="#2563eb", linewidth=1.4, marker="o", markersize=2.4, label="running reqs sum")
        ax.plot(rel, queue, color="#f97316", linewidth=1.0, marker=".", markersize=2.0, label="queue reqs sum")
        ax.set_title(f"step{step}: original {window_start:.1f}s-{window_end:.1f}s, 0.1s samples")
        ax.set_xlabel("relative time in selected window (s)")
        ax.set_ylabel("request count")
        ax.set_ylim(0, y_limit)
        ax.grid(True, axis="both", alpha=0.25)
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.legend(loc="upper right")

    fig.suptitle(args.title or f"SGLang running requests zoom: {jsonl_path.name}", y=0.995)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180)

    print(f"jsonl={jsonl_path}")
    print(f"all_peak_segments={[(round(s[2], 1), round(s[3], 1), s[4]) for s in all_segments]}")
    print(f"merged_peak_segments={[(round(s[2], 1), round(s[3], 1), s[4]) for s in merged_segments]}")
    print(f"main_peak_segments={[(round(s[2], 1), round(s[3], 1), s[4]) for s in main_segments]}")
    print(f"png={output_png}")
    for step, window_start, window_end, _max_running, sub_points in windows:
        running = [point[1] for point in sub_points]
        print(
            f"step{step} window={window_start:.1f},{window_end:.1f} points={len(sub_points)} "
            f"running_max={max(running)} running_zero_points={sum(value == 0 for value in running)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
