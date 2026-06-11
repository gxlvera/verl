"""HotpotQA online search tool for veRL function-tool rollout."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from verl.tools.function_tool import function_tool


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# --- Simulated online-search (SerpAPI) latency on top of the real BM25 retrieve. --
# BM25 supplies real content (~ms), but to mimic a remote search API we additionally
# sleep a sampled delay. Same env contract as hotpot_fixed_context_tool.py so the
# existing search_latencies.csv (mean ~2.7s) can be reused via HOTPOT_TOOL_LATENCY_CSV.
_LATENCY_CSV_VALUES: list[float] | None = None
_SLEEP_RNG: random.Random | None = None
_SLEEP_RNG_SEED: str | None = None


def _sleep_rng() -> random.Random:
    global _SLEEP_RNG, _SLEEP_RNG_SEED
    seed = _env("HOTPOT_TOOL_SLEEP_SEED")
    if _SLEEP_RNG is None or seed != _SLEEP_RNG_SEED:
        _SLEEP_RNG_SEED = seed
        _SLEEP_RNG = random.Random(int(seed)) if seed else random.Random()
    return _SLEEP_RNG


def _load_latency_csv(path: str) -> list[float]:
    global _LATENCY_CSV_VALUES
    if _LATENCY_CSV_VALUES is not None:
        return _LATENCY_CSV_VALUES
    import csv as _csv

    col = _env("HOTPOT_TOOL_LATENCY_CSV_COLUMN") or "search_elapsed_s"
    values: list[float] = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            try:
                values.append(float(row[col]))
            except (KeyError, TypeError, ValueError):
                continue
    _LATENCY_CSV_VALUES = values
    return values


def _sample_sleep_ms() -> float:
    csv_path = _env("HOTPOT_TOOL_LATENCY_CSV")
    if csv_path:
        values = _load_latency_csv(csv_path)
        if values:
            return _sleep_rng().choice(values) * 1000.0
    fixed = _env("HOTPOT_TOOL_SLEEP_MS")
    return float(fixed) if fixed else 0.0


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset, errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from retrieval service: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Retrieval service request failed: {exc.reason}") from exc


def _document_text(document: Any) -> str:
    if isinstance(document, str):
        return document.strip()
    if isinstance(document, dict):
        title = str(document.get("title") or "").strip()
        contents = str(document.get("contents") or document.get("text") or document.get("snippet") or "").strip()
        url = str(document.get("url") or "").strip()
        parts = []
        if title and title not in contents:
            parts.append(title)
        if contents:
            parts.append(contents)
        if url and url not in contents:
            parts.append(f"URL: {url}")
        return "\n".join(parts).strip()
    return str(document).strip()


def _format_results(query: str, payload: dict[str, Any], max_chars: int) -> str:
    result = payload.get("result") or []
    hits = result[0] if result and isinstance(result[0], list) else []
    if not hits:
        return f"<information>\nNo web search results found for query: {query}\n</information>"

    blocks = []
    for idx, hit in enumerate(hits, start=1):
        document = hit.get("document") if isinstance(hit, dict) else hit
        text = _document_text(document)
        if not text:
            continue
        score = hit.get("score") if isinstance(hit, dict) else None
        heading = f"[{idx}]"
        if score is not None:
            heading = f"{heading} score={score}"
        blocks.append(f"{heading}\n{text}")

    content = "\n\n".join(blocks) or f"No readable web search results found for query: {query}"
    wrapped = f"<information>\n{content}\n</information>"
    if max_chars > 0 and len(wrapped) > max_chars:
        wrapped = wrapped[: max_chars - len("\n</information>")] + "\n</information>"
    return wrapped


_RETRIEVE_HOTPOT_CONTEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_hotpot_context",
        "description": (
            "Search the web for evidence relevant to a HotpotQA question. "
            "Use focused entity or relation queries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused web search query for the missing evidence.",
                },
                "answer": {
                    "type": "string",
                    "description": "Optional current answer draft; accepted for automatic tool calls.",
                },
            },
            "required": ["query"],
        },
    },
}


@function_tool("retrieve_hotpot_context", schema=_RETRIEVE_HOTPOT_CONTEXT_SCHEMA)
async def retrieve_hotpot_context(query: str | None = None, answer: str | None = None) -> tuple[str, float, dict[str, Any]]:
    """Retrieve online evidence for a HotpotQA rollout.

    Args:
        query: Focused web search query generated by the model.
        answer: Optional answer draft passed by automatic tool-call forcing.
    """
    del answer
    search_query = (query or "").strip()
    if not search_query:
        return "<information>\nEmpty search query.\n</information>", 0.0, {"search_error": "empty_query"}

    service_url = _env("ONLINE_SEARCH_RETRIEVAL_URL", _env("ONLINE_SEARCH_URL", "http://127.0.0.1:8000/retrieve"))
    topk = int(_env("ONLINE_SEARCH_TOPK", "3"))
    timeout = float(_env("ONLINE_SEARCH_TIMEOUT", "20"))
    max_chars = int(_env("ONLINE_SEARCH_MAX_CHARS", _env("MAX_TOOL_RESPONSE_LENGTH", "12000")))
    started = time.perf_counter()
    payload = {"queries": [search_query], "topk": topk, "return_scores": True}

    try:
        response = await asyncio.to_thread(_post_json, service_url, payload, timeout)
    except Exception as exc:
        text = f"<information>\nSearch error for query '{search_query}': {exc}\n</information>"
        return text, 0.0, {"search_error": type(exc).__name__, "latency_s": time.perf_counter() - started}

    # Real BM25 content is now in hand; add the simulated SerpAPI latency to mimic a
    # remote online-search API. On a speculative-hit turn this whole tool call is
    # launched early (overlapping the thinking decode), so the sleep is hidden.
    sim_sleep_ms = _sample_sleep_ms()
    if sim_sleep_ms > 0:
        await asyncio.sleep(sim_sleep_ms / 1000.0)

    return (
        _format_results(search_query, response, max_chars),
        0.0,
        {
            "search_error": "",
            "search_topk": topk,
            "simulated_latency_s": sim_sleep_ms / 1000.0,
            "latency_s": time.perf_counter() - started,
        },
    )
