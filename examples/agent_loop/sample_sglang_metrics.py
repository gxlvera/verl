#!/usr/bin/env python3
import argparse
import ast
import json
import re
import signal
import time
from pathlib import Path

import requests
import yaml


DEFAULT_PATTERNS = (
    "num_running_reqs",
    "num_queue_reqs",
    "gen_throughput",
    "decode_sum_seq_lens",
)

_STOP = False


def _handle_stop(signum, frame):
    del signum, frame
    global _STOP
    _STOP = True


def _normalize_target(target: str) -> str:
    target = target.strip()
    if target.startswith("http://") or target.startswith("https://"):
        return target.rstrip("/")
    return f"http://{target}".rstrip("/")


def _targets_from_prometheus_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return []

    targets = []
    for scrape_config in data.get("scrape_configs", []):
        if scrape_config.get("job_name") != "rollout":
            continue
        for static_config in scrape_config.get("static_configs", []):
            targets.extend(static_config.get("targets", []))
    return [_normalize_target(target) for target in targets]


def _targets_from_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return []

    matches = re.findall(r"LLMServerManager:\s*(\[[^\n\r]*\])", text)
    if not matches:
        return []
    try:
        targets = ast.literal_eval(matches[-1])
    except Exception:
        return []
    return [_normalize_target(str(target)) for target in targets]


def discover_targets(args) -> list[str]:
    targets = []
    targets.extend(_normalize_target(target) for target in args.target)
    if args.log_file:
        targets.extend(_targets_from_log(Path(args.log_file)))
    if args.prometheus_file and not targets:
        targets.extend(_targets_from_prometheus_file(Path(args.prometheus_file)))

    deduped = []
    seen = set()
    for target in targets:
        if target not in seen:
            deduped.append(target)
            seen.add(target)
    return deduped


def parse_metrics(text: str, patterns: tuple[str, ...]) -> dict:
    series = {}
    summary = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, value_text = line.rpartition(" ")
        if not head or not value_text:
            continue
        metric_name = head.split("{", 1)[0]
        if not any(pattern in metric_name for pattern in patterns):
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        series[head] = value
        bucket = summary.setdefault(metric_name, {"values": [], "sum": 0.0, "max": value, "last": value})
        bucket["values"].append(value)
        bucket["sum"] += value
        bucket["max"] = max(bucket["max"], value)
        bucket["last"] = value

    for bucket in summary.values():
        values = bucket["values"]
        bucket["count"] = len(values)
        bucket["mean"] = bucket["sum"] / len(values) if values else 0.0
        del bucket["values"]
    return {"series": series, "summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample SGLang /metrics at a fixed interval and write JSONL.")
    parser.add_argument("--target", action="append", default=[], help="SGLang target, e.g. 10.0.0.1:34567")
    parser.add_argument("--log-file", default="", help="Run log containing the LLMServerManager target line")
    parser.add_argument(
        "--prometheus-file",
        default="/tmp/ray/session_latest/metrics/prometheus/prometheus.yml",
        help="Prometheus config written by VERL when rollout prometheus is enabled",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--interval", type=float, default=0.1, help="Sampling interval in seconds")
    parser.add_argument("--wait-timeout", type=float, default=900.0, help="Seconds to wait for target discovery")
    parser.add_argument("--request-timeout", type=float, default=0.25, help="HTTP request timeout in seconds")
    parser.add_argument("--pattern", action="append", default=[], help="Metric name substring to keep")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_PATTERNS
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    targets = []
    with output_path.open("a", buffering=1) as fout:
        while not _STOP:
            targets = discover_targets(args)
            if targets:
                fout.write(
                    json.dumps(
                        {"event": "targets_discovered", "ts": time.time(), "targets": targets},
                        sort_keys=True,
                    )
                    + "\n"
                )
                break
            if time.monotonic() - start > args.wait_timeout:
                fout.write(
                    json.dumps(
                        {"event": "target_discovery_timeout", "ts": time.time(), "wait_timeout": args.wait_timeout},
                        sort_keys=True,
                    )
                    + "\n"
                )
                return 2
            time.sleep(min(args.interval, 0.5))

        next_tick = time.monotonic()
        while not _STOP:
            scrape_started = time.monotonic()
            refreshed_targets = discover_targets(args)
            if refreshed_targets and refreshed_targets != targets:
                targets = refreshed_targets
                fout.write(
                    json.dumps(
                        {"event": "targets_refreshed", "ts": time.time(), "targets": targets},
                        sort_keys=True,
                    )
                    + "\n"
                )
            for target in targets:
                url = f"{target}/metrics"
                record = {"ts": time.time(), "monotonic_s": scrape_started - start, "target": target}
                try:
                    response = requests.get(url, timeout=args.request_timeout)
                    record["status_code"] = response.status_code
                    if response.ok:
                        record.update(parse_metrics(response.text, patterns))
                    else:
                        record["error"] = response.text[:200]
                except Exception as exc:
                    record["error"] = repr(exc)
                record["fetch_s"] = time.monotonic() - scrape_started
                fout.write(json.dumps(record, sort_keys=True) + "\n")

            next_tick += args.interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
