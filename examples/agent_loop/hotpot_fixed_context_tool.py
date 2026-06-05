# Copyright 2026
"""HotpotQA fixed-context tool used for rollout latency profiling."""

from __future__ import annotations

import asyncio
import os
import random

from verl.tools.function_tool import function_tool

_SLEEP_RNG: random.Random | None = None
_SLEEP_RNG_SEED: str | None = None

_FIXED_CONTEXT_500_TOKENS = " ".join(
    f"evidence_{idx:03d}" for idx in range(500)
)


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


def _sample_sleep_ms() -> float:
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
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000.0)
    return _FIXED_CONTEXT_500_TOKENS
