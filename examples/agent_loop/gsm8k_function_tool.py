# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GSM8K ``calc_gsm8k_reward`` tool for the ``tool_agent`` agent loop.

This fork replaced the legacy ``BaseTool`` + yaml mechanism with the lightweight
``@function_tool`` registration (``verl/tools/function_tool.py``). Point the
rollout at this file via::

    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/agent_loop/gsm8k_function_tool.py

The tool receives the dataset ground truth through hidden runtime kwargs, so the
model only sees the ``answer`` argument while the tool can still tell it whether
the current candidate is correct. The *training* reward still comes from the
reward manager via ``reward_model.ground_truth`` in the dataset.

Latency knob for profiling
---------------------------
Set ``GSM8K_TOOL_SLEEP_MS`` to inject artificial per-call latency, emulating the
high-latency tools (web search, code sandbox, remote APIs) that motivate the
draft-model tool-call prefetch work. Set ``GSM8K_TOOL_SLEEP_DIST`` for a random
weighted latency distribution, e.g. ``0:30,2000:70`` means 30% no sleep and 70%
2000ms sleep. Set ``GSM8K_TOOL_SLEEP_SEED`` to make the random distribution
repeatable within each tool worker process. The distribution overrides the fixed
sleep. If neither env var is set, or the fixed sleep is 0, no sleep is injected.
"""

from __future__ import annotations

import asyncio
import os
import random
import re

from verl.tools.function_tool import function_tool

_SLEEP_RNG: random.Random | None = None
_SLEEP_RNG_SEED: str | None = None


def _extract_number(text: str) -> str | None:
    """Return the last number-like token in ``text`` (commas stripped), or None."""
    matches = re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    parsed = _extract_number(str(answer))
    if parsed is None:
        return None
    try:
        value = float(parsed)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except ValueError:
        return parsed


def _parse_sleep_distribution(spec: str) -> list[tuple[float, float]]:
    """Parse ``sleep_ms:weight`` pairs, e.g. ``0:30,2000:70``."""
    choices: list[tuple[float, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "GSM8K_TOOL_SLEEP_DIST must use 'sleep_ms:weight' pairs, "
                "for example '0:30,2000:70'."
            )
        sleep_raw, weight_raw = item.split(":", 1)
        sleep_ms = float(sleep_raw.strip())
        weight = float(weight_raw.strip().rstrip("%"))
        if sleep_ms < 0:
            raise ValueError("GSM8K_TOOL_SLEEP_DIST sleep values must be non-negative.")
        if weight <= 0:
            raise ValueError("GSM8K_TOOL_SLEEP_DIST weights must be positive.")
        choices.append((sleep_ms, weight))
    if not choices:
        raise ValueError("GSM8K_TOOL_SLEEP_DIST is set but contains no choices.")
    return choices


def _sleep_rng() -> random.Random:
    """Return a process-local RNG for artificial tool latency sampling."""
    global _SLEEP_RNG, _SLEEP_RNG_SEED
    seed = os.getenv("GSM8K_TOOL_SLEEP_SEED", "").strip()
    if _SLEEP_RNG is None or seed != _SLEEP_RNG_SEED:
        _SLEEP_RNG_SEED = seed
        _SLEEP_RNG = random.Random(int(seed)) if seed else random.Random()
    return _SLEEP_RNG


def _sample_sleep_ms() -> float:
    dist_spec = os.getenv("GSM8K_TOOL_SLEEP_DIST", "").strip()
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

    fixed_sleep = os.getenv("GSM8K_TOOL_SLEEP_MS", "").strip()
    return float(fixed_sleep) if fixed_sleep else 0.0


_CALC_GSM8K_REWARD_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calc_gsm8k_reward",
        "description": (
            "Check a candidate GSM8K numeric answer. Use this tool after you have "
            "worked through the problem and have a candidate answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Your current candidate answer, ideally formatted as `#### <number>`.",
                }
            },
            "required": ["answer"],
        },
    },
}


@function_tool("calc_gsm8k_reward", schema=_CALC_GSM8K_REWARD_SCHEMA)
async def calc_gsm8k_reward(answer: str, ground_truth: str | None = None) -> str:
    """Validate a candidate GSM8K answer and return formatting feedback.

    Call this once you have worked through the problem step by step and have a
    candidate numeric answer. It checks whether your candidate matches the
    hidden ground truth when available.

    Args:
        answer: Your current candidate answer, ideally already formatted as
            ``#### <number>`` (for example "#### 42").
        ground_truth: Hidden runtime ground truth injected by the rollout system.
    """
    # Optional artificial latency to emulate high-latency tools for rollout
    # profiling. This is an async sleep so other in-flight trajectories can
    # continue running while this tool call is waiting.
    sleep_ms = _sample_sleep_ms()
    if sleep_ms > 0:
        await asyncio.sleep(sleep_ms / 1000.0)

    parsed = _normalize_answer(answer)
    if parsed is None:
        return (
            "Could not parse a number from your answer. Make sure your final "
            "answer is in the format `#### <number>`, e.g. `#### 42`."
        )
    expected = _normalize_answer(ground_truth)
    if expected is None:
        return f"Parsed answer: {parsed}. If you are confident, output it as `#### {parsed}`."
    if parsed == expected:
        return f"Correct. Your answer {parsed} matches the ground truth. You may output `#### {parsed}`."
    return (
        f"Incorrect. Your current answer {parsed} does not match the ground truth. "
        "Please rethink the solution carefully and call this tool again with a revised numeric answer."
    )
