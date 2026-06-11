# Copyright 2026
"""HotpotQA fixed-context tool used for rollout latency profiling."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import random

from verl.tools.function_tool import function_tool

logger = logging.getLogger(__name__)

_SLEEP_RNG: random.Random | None = None
_SLEEP_RNG_SEED: str | None = None

# Empirical latency samples loaded from a CSV column (e.g. measured serpapi search
# latencies). Cached after first load; each tool call bootstrap-samples one value.
_LATENCY_CSV_VALUES: list[float] | None = None

# Target size (in model tokens) of the synthetic evidence passage returned by the
# tool on every call. Defaults to 500 tokens. The text is sized with the model
# tokenizer so it re-encodes to ~this many tokens (see _fixed_context).
_RESPONSE_TOKENS = int(os.getenv("HOTPOT_TOOL_RESPONSE_TOKENS", "500") or "500")

_TOKENIZER = None
_TOKENIZER_DISABLED = False


def _get_tokenizer():
    """Lazily load the model tokenizer used to size tool responses by token count.

    Path resolution: HOTPOT_TOOL_TOKENIZER, then MODEL_PATH. Returns None if no
    path is configured or loading fails, in which case _fixed_context falls back
    to a word-count approximation.
    """
    global _TOKENIZER, _TOKENIZER_DISABLED
    if _TOKENIZER is not None or _TOKENIZER_DISABLED:
        return _TOKENIZER
    path = (os.getenv("HOTPOT_TOOL_TOKENIZER") or os.getenv("MODEL_PATH") or "").strip()
    if not path:
        _TOKENIZER_DISABLED = True
        logger.warning("HOTPOT_TOOL_TOKENIZER/MODEL_PATH unset; tool response sized by word count.")
        return None
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        _TOKENIZER_DISABLED = True
        logger.warning("Failed to load tokenizer from %s (%r); sizing tool response by word count.", path, exc)
    return _TOKENIZER


def _parse_sleep_distribution(spec: str) -> list[tuple[float, float]]:
    choices: list[tuple[float, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("HOTPOT_TOOL_SLEEP_DIST must use 'sleep_ms:weight' pairs.")
        sleep_raw, weight_raw = item.split(":", 1)
        sleep_ms = float(sleep_raw.strip())
        weight = float(weight_raw.strip().rstrip("%"))
        if sleep_ms < 0:
            raise ValueError("HOTPOT_TOOL_SLEEP_DIST sleep values must be non-negative.")
        if weight <= 0:
            raise ValueError("HOTPOT_TOOL_SLEEP_DIST weights must be positive.")
        choices.append((sleep_ms, weight))
    if not choices:
        raise ValueError("HOTPOT_TOOL_SLEEP_DIST is set but contains no choices.")
    return choices


def _sleep_rng() -> random.Random:
    global _SLEEP_RNG, _SLEEP_RNG_SEED
    seed = os.getenv("HOTPOT_TOOL_SLEEP_SEED", os.getenv("GSM8K_TOOL_SLEEP_SEED", "")).strip()
    if _SLEEP_RNG is None or seed != _SLEEP_RNG_SEED:
        _SLEEP_RNG_SEED = seed
        _SLEEP_RNG = random.Random(int(seed)) if seed else random.Random()
    return _SLEEP_RNG


def _load_latency_csv(path: str) -> list[float]:
    """Load (and cache) a column of latency-in-seconds values from a CSV file.

    Column defaults to 'search_elapsed_s' (the serpapi search latency column from
    the 100-call profiling run); override via HOTPOT_TOOL_LATENCY_CSV_COLUMN.
    """
    global _LATENCY_CSV_VALUES
    if _LATENCY_CSV_VALUES is not None:
        return _LATENCY_CSV_VALUES
    import csv as _csv

    col = os.getenv("HOTPOT_TOOL_LATENCY_CSV_COLUMN", "search_elapsed_s").strip()
    values: list[float] = []
    with open(path, newline="") as f:
        for row in _csv.DictReader(f):
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue
    if not values:
        raise ValueError(f"HOTPOT_TOOL_LATENCY_CSV {path!r} has no values in column {col!r}.")
    _LATENCY_CSV_VALUES = values
    logger.info("Loaded %d latency samples from %s column %s.", len(values), path, col)
    return values


def _sample_sleep_ms() -> float:
    csv_path = os.getenv("HOTPOT_TOOL_LATENCY_CSV", "").strip()
    if csv_path:
        values = _load_latency_csv(csv_path)
        return _sleep_rng().choice(values) * 1000.0

    seconds_list = os.getenv("HOTPOT_TOOL_LATENCY_SECONDS_LIST", "").strip()
    if seconds_list:
        values = [float(item.strip()) for item in seconds_list.split(",") if item.strip()]
        if not values:
            raise ValueError("HOTPOT_TOOL_LATENCY_SECONDS_LIST is set but contains no values.")
        return _sleep_rng().choice(values) * 1000.0

    dist_spec = os.getenv("HOTPOT_TOOL_SLEEP_DIST", os.getenv("GSM8K_TOOL_SLEEP_DIST", "")).strip()
    if dist_spec:
        choices = _parse_sleep_distribution(dist_spec)
        total_weight = sum(weight for _, weight in choices)
        draw = _sleep_rng().uniform(0.0, total_weight)
        cumulative = 0.0
        for sleep_ms, weight in choices:
            cumulative += weight
            if draw <= cumulative:
                return sleep_ms
        return choices[-1][0]

    fixed_sleep = os.getenv("HOTPOT_TOOL_SLEEP_MS", os.getenv("GSM8K_TOOL_SLEEP_MS", "")).strip()
    return float(fixed_sleep) if fixed_sleep else 0.0


@functools.lru_cache(maxsize=256)
def _fixed_context(query: str | None, answer: str | None) -> str:
    """Return a synthetic evidence passage of ~_RESPONSE_TOKENS model tokens.

    The passage is deterministic and unique per (query, answer) so prefix caching
    does not collapse different questions. When the tokenizer is available the
    text is trimmed to exactly _RESPONSE_TOKENS tokens; otherwise it falls back to
    a word-count approximation.
    """
    key = f"{query or ''}\n{answer or ''}".encode("utf-8", errors="replace")
    digest = hashlib.sha1(key).hexdigest()[:10]
    target = _RESPONSE_TOKENS

    tokenizer = _get_tokenizer()
    if tokenizer is not None:
        # Build a unique filler that is comfortably longer than the target, then
        # trim to exactly `target` token ids and decode back to text.
        filler = " ".join(f"evidence_{digest}_{idx:04d}" for idx in range(target + 64))
        token_ids = tokenizer(filler, add_special_tokens=False)["input_ids"][:target]
        return tokenizer.decode(token_ids)

    # Fallback: no tokenizer available; approximate by word count.
    return " ".join(f"evidence_{digest}_{idx:04d}" for idx in range(target))


_RETRIEVE_HOTPOT_CONTEXT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_hotpot_context",
        "description": (
            "Retrieve a fixed evidence passage for the current HotpotQA question. "
            "Use this tool before answering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short search query or current reasoning state.",
                },
                "answer": {
                    "type": "string",
                    "description": "Optional current answer draft; accepted for automatic tool calls.",
                },
            },
        },
    },
}


@function_tool("retrieve_hotpot_context", schema=_RETRIEVE_HOTPOT_CONTEXT_SCHEMA)
async def retrieve_hotpot_context(query: str | None = None, answer: str | None = None) -> str:
    """Return precomputed fixed context for HotpotQA profiling.

    Args:
        query: Optional search query from the model.
        answer: Optional answer draft passed by automatic tool-call forcing.
    """
    sleep_ms = _sample_sleep_ms()
    started = asyncio.get_running_loop().time()
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000.0)
    latency_s = asyncio.get_running_loop().time() - started
    return (
        _fixed_context(query, answer),
        0.0,
        {
            "configured_sleep_s": sleep_ms / 1000.0,
            "latency_s": latency_s,
            "query": query or "",
        },
    )
