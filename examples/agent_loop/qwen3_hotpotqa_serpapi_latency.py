#!/usr/bin/env python3
"""Generate HotpotQA search tool calls with Qwen3-8B and time SerpApi calls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from openai import OpenAI

import qwen3_tau_toolcall_eval as common


SEARCH_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for evidence needed to answer a HotpotQA question.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30003)
    parser.add_argument("--dataset-path", default="/root/data/hotpotqa_tool_agent/test.parquet")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="runs/qwen3_8b_hotpotqa_serpapi_latency_100")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--server-timeout-s", type=int, default=900)
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--tool-call-parser", default="qwen25")
    parser.add_argument("--tool-choice", default="required", choices=["auto", "required"])
    parser.add_argument("--no-launch-server", action="store_true")
    parser.add_argument("--sglang-arg", action="append", default=[])
    parser.add_argument("--non-stream", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--serpapi-env", default="/root/search_api.env")
    parser.add_argument("--serpapi-url", default="https://serpapi.com/search.json")
    parser.add_argument("--serpapi-num", type=int, default=5)
    parser.add_argument("--serpapi-timeout-s", type=float, default=60.0)
    return parser.parse_args()


def load_serpapi_key(env_path: str) -> str:
    key = os.environ.get("SERPAPI_API_KEY", "").strip()
    if key:
        return key
    path = Path(env_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("SERPAPI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("SERPAPI_API_KEY is not set and was not found in the env file.")


def build_messages(question: str) -> list[dict[str, str]]:
    system = (
        "You are a HotpotQA search planner. Given a question, make exactly one search_web "
        "tool call with a concise query that is likely to retrieve evidence for answering it. "
        "Do not answer the question in text."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def build_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    df = pd.read_parquet(args.dataset_path)
    if args.max_examples > 0:
        df = df.sample(n=min(args.max_examples, len(df)), random_state=args.seed)
    rows = df.reset_index(drop=False).to_dict("records")
    items: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = str(row["question"])
        items.append(
            {
                "record_id": str(row.get("id") or f"hotpotqa-{row['index']}"),
                "task_id": int(row["index"]),
                "turn_index": 0,
                "messages": build_messages(question),
                "tools": SEARCH_TOOL,
                "target_tool_calls": [],
                "metadata": {"question": question, "answer": row.get("answer")},
                "sample_index": idx,
            }
        )
    return items


def first_query(tool_calls: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    if not tool_calls:
        return None, None
    call = common.normalize_tool_call(tool_calls[0])
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    query = str(args.get("query") or "").strip()
    return (query or None), call


def run_serpapi(query: str, api_key: str, args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=args.serpapi_timeout_s) as client:
            response = client.get(
                args.serpapi_url,
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": api_key,
                    "num": args.serpapi_num,
                },
            )
        elapsed = time.perf_counter() - start
        response.raise_for_status()
        payload = response.json()
        return {
            "ok": True,
            "elapsed_s": elapsed,
            "status_code": response.status_code,
            "result_count": len(payload.get("organic_results") or []),
            "search_metadata": payload.get("search_metadata") or {},
            "organic_results": (payload.get("organic_results") or [])[: args.serpapi_num],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "elapsed_s": time.perf_counter() - start, "error": repr(exc)}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean_s": statistics.fmean(values) if values else None,
        "stdev_s": statistics.stdev(values) if len(values) > 1 else None,
        "min_s": min(values) if values else None,
        "p50_s": percentile(values, 50),
        "p75_s": percentile(values, 75),
        "p90_s": percentile(values, 90),
        "p95_s": percentile(values, 95),
        "p99_s": percentile(values, 99),
        "max_s": max(values) if values else None,
    }


def write_latency_csv(records: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "sample_index",
        "record_id",
        "question",
        "query",
        "model_ok",
        "model_elapsed_s",
        "model_completion_tokens",
        "search_ok",
        "search_elapsed_s",
        "search_result_count",
        "search_error",
    ]
    with (out_dir / "search_latencies.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            search = record.get("search") or {}
            model = record.get("model") or {}
            writer.writerow(
                {
                    "sample_index": record.get("sample_index"),
                    "record_id": record.get("record_id"),
                    "question": record.get("question"),
                    "query": record.get("query"),
                    "model_ok": model.get("ok"),
                    "model_elapsed_s": model.get("elapsed_s"),
                    "model_completion_tokens": model.get("completion_tokens"),
                    "search_ok": search.get("ok"),
                    "search_elapsed_s": search.get("elapsed_s"),
                    "search_result_count": search.get("result_count"),
                    "search_error": search.get("error"),
                }
            )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = load_serpapi_key(args.serpapi_env)
    model = args.served_model_name or args.model_path

    proc = None
    try:
        proc = common.launch_server(args)
        client = OpenAI(api_key="EMPTY", base_url=f"{common.server_url(args)}/v1")
        items = build_items(args)
        records: list[dict[str, Any]] = []
        jsonl_path = out_dir / "records.jsonl"
        enable_thinking = not args.disable_thinking

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for idx, item in enumerate(items, start=1):
                print(f"[{idx}/{len(items)}] {item['record_id']}", flush=True)
                result = common.call_model(client, model, item, enable_thinking, args)
                query, normalized_call = first_query(result.tool_calls)
                model_record = {
                    "ok": result.ok,
                    "elapsed_s": result.elapsed_s,
                    "ttft_s": result.ttft_s,
                    "completion_tokens": result.completion_tokens,
                    "prompt_tokens": result.prompt_tokens,
                    "total_tokens": result.total_tokens,
                    "error": result.error,
                    "output_text": result.output_text,
                    "reasoning_text": result.reasoning_text,
                    "raw_tool_calls": result.tool_calls,
                    "normalized_first_tool_call": normalized_call,
                }
                search_record = run_serpapi(query, api_key, args) if query else {"ok": False, "error": "missing model query"}
                record = {
                    "sample_index": item["sample_index"],
                    "record_id": item["record_id"],
                    "task_id": item["task_id"],
                    "question": item["metadata"]["question"],
                    "answer": item["metadata"]["answer"],
                    "query": query,
                    "model": model_record,
                    "search": search_record,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"  query={query!r} model_elapsed={result.elapsed_s:.3f}s "
                    f"search_ok={search_record.get('ok')} search_elapsed={search_record.get('elapsed_s')}",
                    flush=True,
                )

        search_latencies = [r["search"]["elapsed_s"] for r in records if r.get("search", {}).get("ok")]
        model_latencies = [r["model"]["elapsed_s"] for r in records if r.get("model", {}).get("ok")]
        summary = {
            "model": model,
            "dataset_path": args.dataset_path,
            "requested_examples": args.max_examples,
            "n_records": len(records),
            "n_model_ok": sum(bool(r["model"].get("ok")) for r in records),
            "n_with_query": sum(bool(r.get("query")) for r in records),
            "n_search_ok": sum(bool(r["search"].get("ok")) for r in records),
            "n_search_failed": sum(not bool(r["search"].get("ok")) for r in records),
            "thinking_enabled": enable_thinking,
            "serpapi_num": args.serpapi_num,
            "model_latency": latency_summary(model_latencies),
            "search_latency": latency_summary(search_latencies),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_latency_csv(records, out_dir)
        print(f"Wrote {jsonl_path}", flush=True)
        print(f"Wrote {out_dir / 'summary.json'} and {out_dir / 'search_latencies.csv'}", flush=True)
        print("SEARCH_LATENCY_SUMMARY " + json.dumps(summary["search_latency"], ensure_ascii=False), flush=True)
        return 0
    finally:
        common.stop_server(proc)


if __name__ == "__main__":
    raise SystemExit(main())
