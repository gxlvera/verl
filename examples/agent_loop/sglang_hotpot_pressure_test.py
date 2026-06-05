#!/usr/bin/env python3
"""Run standalone SGLang scheduling pressure tests with HotpotQA prompts."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp
import matplotlib
import requests
from datasets import load_dataset
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_PATTERNS = (
    "num_running_reqs",
    "num_queue_reqs",
    "gen_throughput",
    "decode_sum_seq_lens",
)


def parse_metrics(text: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, value_text = line.rpartition(" ")
        if not head or not value_text:
            continue
        metric_name = head.split("{", 1)[0]
        if not any(pattern in metric_name for pattern in METRIC_PATTERNS):
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        bucket = summary.setdefault(metric_name, {"sum": 0.0, "max": value, "last": value, "count": 0.0})
        bucket["sum"] += value
        bucket["max"] = max(bucket["max"], value)
        bucket["last"] = value
        bucket["count"] += 1.0
    for bucket in summary.values():
        bucket["mean"] = bucket["sum"] / bucket["count"] if bucket["count"] else 0.0
    return summary


def metric_value(summary: dict[str, dict[str, float]], name: str) -> float:
    for metric_name, values in summary.items():
        if name in metric_name:
            return values.get("last", 0.0)
    return 0.0


def make_filler(tokenizer: Any, target_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    snippets = [
        "Observation: the mock tool returned a long intermediate reasoning trace with arithmetic checks.",
        "ToolResult: retrieved evidence contains multiple titles, bridge entities, dates, aliases, and distractors.",
        "Scratchpad: compare the candidate answer against every supporting sentence before finalizing.",
        "VerifierNote: keep the original question in mind and ignore unrelated context fragments.",
    ]
    chunks = []
    while True:
        chunks.append(rng.choice(snippets))
        chunks.append(f" synthetic_call_id={len(chunks)} confidence={rng.random():.4f};")
        text = " ".join(chunks)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            return tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)


def make_filler_ids(tokenizer: Any, target_tokens: int, seed: int) -> list[int]:
    text = make_filler(tokenizer, target_tokens, seed)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:target_tokens]


def context_to_text(context: Any, max_titles: int = 6, max_sentences_per_title: int = 4) -> str:
    titles = context.get("title", []) if isinstance(context, dict) else []
    sentences = context.get("sentences", []) if isinstance(context, dict) else []
    parts = []
    for title, sents in list(zip(titles, sentences))[:max_titles]:
        joined = " ".join(str(s) for s in list(sents)[:max_sentences_per_title])
        parts.append(f"[{title}] {joined}")
    return "\n".join(parts)


def build_prompts(args: argparse.Namespace) -> list[str]:
    dataset = load_dataset("hotpot_qa", "distractor", split="train")
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.num_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    filler = make_filler(tokenizer, args.extra_tokens, args.seed + 17)
    prompts = []
    for request_id, index in enumerate(indices):
        row = dataset[index]
        prompt = (
            "You are answering a HotpotQA multi-hop question. Use the provided context and the long "
            "mock tool transcript. Return a concise final answer.\n\n"
            f"Request id: {request_id}\n"
            f"Question: {row['question']}\n\n"
            f"Context:\n{context_to_text(row['context'])}\n\n"
            f"Mock tool transcript, intentionally long:\n{filler}\n\n"
            "Answer:"
        )
        prompts.append(prompt)
    return prompts


def build_multi_turn_prompt_records(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    dataset = load_dataset("hotpot_qa", "distractor", split="train")
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.num_prompts)
    prompt_records = []
    system_prompt = (
        "System: You are a HotpotQA search agent. Answer by thinking briefly. "
        "If more evidence is needed, emit exactly one tool call in this format: "
        "<tool_call>{\"name\":\"search\",\"arguments\":{\"query\":\"...\"}}</tool_call>. "
        "After a search result is appended, continue from the full conversation. "
        "Do not invent tool outputs; wait for the provided search result.\n"
    )
    reminder_text = (
        " Tool format reminder: use a single JSON search call when evidence is missing. "
        "The search result will be appended by the environment as a tool message. "
    )
    reminder_ids = tokenizer.encode(reminder_text, add_special_tokens=False)
    pad_ids = (reminder_ids * ((args.initial_prompt_tokens // max(1, len(reminder_ids))) + 2))[
        : args.initial_prompt_tokens
    ]
    for request_id, index in enumerate(indices):
        row = dataset[index]
        prefix_ids = tokenizer.encode(system_prompt, add_special_tokens=False)
        suffix_text = f"\nUser: Question: {row['question']}\nAssistant:"
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        if len(prefix_ids) + len(suffix_ids) < args.initial_prompt_tokens:
            need = args.initial_prompt_tokens - len(prefix_ids) - len(suffix_ids)
            base_ids = prefix_ids + pad_ids[:need] + suffix_ids
        else:
            base_ids = prefix_ids + suffix_ids
        prompt_records.append(
            {
                "request_id": request_id,
                "hotpot_index": index,
                "question": row["question"],
                "answer": row["answer"],
                "base_prompt": tokenizer.decode(base_ids, skip_special_tokens=True),
                "base_ids": base_ids,
                "base_prompt_tokens": len(base_ids),
            }
        )
    return prompt_records


def start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        args.model_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--enable-metrics",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--context-length",
        str(args.context_length),
    ]
    if args.disable_cuda_graph_server:
        cmd.append("--disable-cuda-graph")
        cmd.append("--disable-piecewise-cuda-graph")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    log_f = log_path.open("a", buffering=1)
    log_f.write("cmd=" + " ".join(cmd) + "\n")
    log_f.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, text=True)
    proc._sglang_log_file = log_f  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)
    log_f = getattr(proc, "_sglang_log_file", None)
    if log_f is not None:
        log_f.close()


def wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited early with code {proc.returncode}")
        for path in ("/health", "/health_generate", "/metrics"):
            try:
                resp = requests.get(base_url + path, timeout=2)
                if resp.ok:
                    return
                last_error = f"{path}: HTTP {resp.status_code}"
            except Exception as exc:
                last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"SGLang server was not ready within {timeout_s}s; last_error={last_error}")


async def sample_metrics(base_url: str, output: Path, interval: float, stop_event: asyncio.Event) -> None:
    start = time.monotonic()
    async with aiohttp.ClientSession() as session:
        with output.open("a", buffering=1) as fout:
            fout.write(json.dumps({"event": "targets_discovered", "ts": time.time(), "targets": [base_url]}) + "\n")
            next_tick = time.monotonic()
            while not stop_event.is_set():
                now = time.monotonic()
                record: dict[str, Any] = {"ts": time.time(), "monotonic_s": now - start, "target": base_url}
                try:
                    async with session.get(base_url + "/metrics", timeout=aiohttp.ClientTimeout(total=0.5)) as resp:
                        record["status_code"] = resp.status
                        text = await resp.text()
                        if resp.status == 200:
                            record["summary"] = parse_metrics(text)
                        else:
                            record["error"] = text[:200]
                except Exception as exc:
                    record["error"] = repr(exc)
                fout.write(json.dumps(record, sort_keys=True) + "\n")
                next_tick += interval
                await asyncio.sleep(max(0.0, next_tick - time.monotonic()))


async def send_one(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt: str,
    request_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    payload = {
        "model": args.model_path,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": args.max_new_tokens,
    }
    start = time.monotonic()
    out: dict[str, Any] = {"request_id": request_id, "submit_s": start}
    try:
        async with session.post(
            base_url + "/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=args.request_timeout),
        ) as resp:
            text = await resp.text()
            out["status"] = resp.status
            out["latency_s"] = time.monotonic() - start
            if resp.status == 200:
                data = json.loads(text)
                choice = data.get("choices", [{}])[0]
                output_text = choice.get("text", "")
                out["output_text"] = output_text
                out["output_chars"] = len(output_text)
                usage = data.get("usage", {})
                out["prompt_tokens"] = usage.get("prompt_tokens")
                out["completion_tokens"] = usage.get("completion_tokens")
                out["total_tokens"] = usage.get("total_tokens")
            else:
                out["error"] = text[:500]
    except Exception as exc:
        out["latency_s"] = time.monotonic() - start
        out["error"] = repr(exc)
    return out


async def send_all(base_url: str, prompts: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(send_one(session, base_url, prompt, i, args)) for i, prompt in enumerate(prompts)]
        return await asyncio.gather(*tasks)


async def send_multi_turn_one(
    session: aiohttp.ClientSession,
    base_url: str,
    record: dict[str, Any],
    tokenizer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    context_ids = list(record["base_ids"])
    result: dict[str, Any] = {
        "request_id": record["request_id"],
        "hotpot_index": record["hotpot_index"],
        "question": record["question"],
        "base_prompt_tokens": record["base_prompt_tokens"],
        "turns": [],
    }
    request_start = time.monotonic()
    for turn in range(1, args.assistant_turns + 1):
        prompt = tokenizer.decode(context_ids, skip_special_tokens=True)
        turn_result = await send_one(session, base_url, prompt, record["request_id"], args)
        assistant_text = turn_result.get("output_text", "")
        assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        turn_info = {
            "turn": turn,
            "prompt_tokens_by_local_tokenizer": len(context_ids),
            "assistant_tokens_by_local_tokenizer": len(assistant_ids),
            **{k: v for k, v in turn_result.items() if k != "output_text"},
        }
        result["turns"].append(turn_info)
        if turn_result.get("status") != 200:
            result["failed"] = True
            result["latency_s"] = time.monotonic() - request_start
            return result
        context_ids.extend(assistant_ids)
        if turn < args.assistant_turns:
            tool_seed = args.seed + 41 + record["request_id"] * 1009 + turn * 9176
            tool_response_ids = make_filler_ids(tokenizer, args.tool_response_tokens, tool_seed)
            context_ids.extend(tool_response_ids)
            result["turns"][-1]["injected_tool_response_tokens"] = len(tool_response_ids)
            result["turns"][-1]["injected_tool_response_seed"] = tool_seed
    result["failed"] = False
    result["latency_s"] = time.monotonic() - request_start
    result["final_context_tokens_by_local_tokenizer"] = len(context_ids)
    return result


async def send_multi_turn_all(
    base_url: str,
    prompt_records: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(send_multi_turn_one(session, base_url, record, tokenizer, args))
            for record in prompt_records
        ]
        return await asyncio.gather(*tasks)


def plot_metrics(jsonl_path: Path, png_path: Path, csv_path: Path, title: str) -> dict[str, float]:
    points = []
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        summary = row.get("summary")
        if not summary:
            continue
        t = round(float(row["monotonic_s"]), 1)
        running = metric_value(summary, "num_running_reqs")
        queue = metric_value(summary, "num_queue_reqs")
        throughput = metric_value(summary, "gen_throughput")
        seq_lens = metric_value(summary, "decode_sum_seq_lens")
        points.append((t, running, queue, throughput, seq_lens))
    if not points:
        raise RuntimeError(f"No metric points found in {jsonl_path}")

    active = [i for i, p in enumerate(points) if p[1] > 0 or p[2] > 0 or p[3] > 0 or p[4] > 0]
    if active:
        first = max(0, active[0] - 10)
        last = min(len(points) - 1, active[-1] + 10)
        points = points[first : last + 1]

    with csv_path.open("w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["monotonic_s", "running_reqs", "queue_reqs", "gen_throughput", "decode_sum_seq_lens"])
        writer.writerows(points)

    xs = [p[0] for p in points]
    running = [p[1] for p in points]
    queue = [p[2] for p in points]
    throughput = [p[3] for p in points]

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=180)
    ax1.plot(xs, running, color="#2563eb", linewidth=1.6, label="running requests")
    ax1.plot(xs, queue, color="#f97316", linewidth=1.4, label="queue requests")
    ax1.set_xlabel("sampler monotonic time (s)")
    ax1.set_ylabel("request count")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis="y", alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, throughput, color="#16a34a", linewidth=1.0, alpha=0.7, label="gen throughput")
    ax2.set_ylabel("gen throughput")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close(fig)

    return {
        "points": float(len(points)),
        "running_max": max(running),
        "queue_max": max(queue),
        "queue_nonzero_points": float(sum(v > 0 for v in queue)),
        "running_zero_points": float(sum(v == 0 for v in running)),
        "throughput_max": max(throughput),
    }


async def run_test(args: argparse.Namespace) -> None:
    run_name = args.run_name or f"hotpot_sglang_pressure_qwen3_8b_gpu{args.gpu}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / run_name
    server_log = base.with_name(base.name + "_server.log")
    prompts_json = base.with_name(base.name + "_prompts.jsonl")
    results_json = base.with_name(base.name + "_results.jsonl")
    metrics_json = base.with_name(base.name + "_sglang_metrics.jsonl")
    plot_png = base.with_name(base.name + "_schedule.png")
    plot_csv = base.with_name(base.name + "_schedule.csv")
    summary_json = base.with_name(base.name + "_summary.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts: list[str] = []
    prompt_records: list[dict[str, Any]] = []
    if args.mode == "single":
        prompts = build_prompts(args)
        with prompts_json.open("w") as fout:
            for i, prompt in enumerate(prompts):
                fout.write(
                    json.dumps({"request_id": i, "prompt_tokens": len(tokenizer.encode(prompt)), "prompt": prompt})
                    + "\n"
                )
    else:
        prompt_records = build_multi_turn_prompt_records(args, tokenizer)
        with prompts_json.open("w") as fout:
            for record in prompt_records:
                fout.write(
                    json.dumps(
                        {
                            "request_id": record["request_id"],
                            "hotpot_index": record["hotpot_index"],
                            "question": record["question"],
                            "base_prompt_tokens": record["base_prompt_tokens"],
                            "base_prompt": record["base_prompt"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    base_url = f"http://{args.host}:{args.port}"
    proc = None
    if args.start_server:
        proc = start_server(args, server_log)
        wait_ready(base_url, proc, args.server_ready_timeout)
    else:
        wait_ready(base_url, subprocess.Popen(["true"]), args.server_ready_timeout)

    stop_event = asyncio.Event()
    sampler = asyncio.create_task(sample_metrics(base_url, metrics_json, args.metrics_interval, stop_event))
    await asyncio.sleep(args.pre_request_sample_s)
    start = time.monotonic()
    if args.mode == "single":
        results = await send_all(base_url, prompts, args)
    else:
        results = await send_multi_turn_all(base_url, prompt_records, tokenizer, args)
    wall_s = time.monotonic() - start
    await asyncio.sleep(args.post_request_sample_s)
    stop_event.set()
    await sampler

    with results_json.open("w") as fout:
        for result in results:
            fout.write(json.dumps(result, sort_keys=True) + "\n")

    plot_stats = plot_metrics(
        metrics_json,
        plot_png,
        plot_csv,
        (
            f"SGLang schedule pressure test: {args.num_prompts} HotpotQA prompts, "
            f"mode={args.mode}, max_new={args.max_new_tokens}"
        ),
    )
    if args.mode == "single":
        ok = sum(1 for r in results if r.get("status") == 200)
    else:
        ok = sum(1 for r in results if not r.get("failed"))
    latencies = [r["latency_s"] for r in results if "latency_s" in r]
    total_turns = 0
    ok_turns = 0
    if args.mode != "single":
        for result in results:
            for turn in result.get("turns", []):
                total_turns += 1
                if turn.get("status") == 200:
                    ok_turns += 1
    summary = {
        "run_name": run_name,
        "base_url": base_url,
        "gpu": args.gpu,
        "mode": args.mode,
        "num_prompts": args.num_prompts,
        "extra_tokens": args.extra_tokens,
        "initial_prompt_tokens": args.initial_prompt_tokens,
        "tool_response_tokens": args.tool_response_tokens,
        "assistant_turns": args.assistant_turns,
        "max_new_tokens": args.max_new_tokens,
        "ok_requests": ok,
        "failed_requests": len(results) - ok,
        "ok_turns": ok_turns,
        "total_turns": total_turns,
        "wall_s": wall_s,
        "latency_min_s": min(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "latency_mean_s": sum(latencies) / len(latencies) if latencies else None,
        "server_log": str(server_log),
        "prompts_jsonl": str(prompts_json),
        "results_jsonl": str(results_json),
        "metrics_jsonl": str(metrics_json),
        "plot_png": str(plot_png),
        "plot_csv": str(plot_csv),
        **plot_stats,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.stop_server and proc is not None:
        stop_server(proc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31002)
    parser.add_argument("--output-dir", default="/root/logs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--mode", choices=["single", "multi_turn_search"], default="multi_turn_search")
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--extra-tokens", type=int, default=2000)
    parser.add_argument("--initial-prompt-tokens", type=int, default=300)
    parser.add_argument("--tool-response-tokens", type=int, default=500)
    parser.add_argument("--assistant-turns", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-interval", type=float, default=0.1)
    parser.add_argument("--pre-request-sample-s", type=float, default=2.0)
    parser.add_argument("--post-request-sample-s", type=float, default=8.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--server-ready-timeout", type=float, default=900.0)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--disable-cuda-graph-server", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-server", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    asyncio.run(run_test(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
