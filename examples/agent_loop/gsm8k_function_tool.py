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

Note: function tools are stateless and intentionally ignore per-trajectory
``tools_kwargs`` (including ``ground_truth``). The *training* reward still comes
from the reward manager via ``reward_model.ground_truth`` in the dataset, so this
in-loop tool only needs to give the model formatting feedback and exercise the
tool-call path. That is exactly what the first-stage rollout profiling needs.

Latency knob for profiling
---------------------------
Set ``GSM8K_TOOL_SLEEP_MS`` to inject artificial per-call latency, emulating the
high-latency tools (web search, code sandbox, remote APIs) that motivate the
draft-model tool-call prefetch work. Defaults to 0 = honest local timing.
"""

from __future__ import annotations

import os
import re
import time

from verl.tools.function_tool import function_tool


def _extract_number(text: str) -> str | None:
    """Return the last number-like token in ``text`` (commas stripped), or None."""
    matches = re.findall(r"-?[0-9][0-9,]*\.?[0-9]*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


@function_tool("calc_gsm8k_reward")
def calc_gsm8k_reward(answer: str) -> str:
    """Validate a candidate GSM8K answer and return formatting feedback.

    Call this once you have worked through the problem step by step and have a
    candidate numeric answer. It checks that the answer is in the required
    ``#### <number>`` format and echoes back the parsed value so you can refine
    it before emitting the final answer.

    Args:
        answer: Your current candidate answer, ideally already formatted as
            ``#### <number>`` (for example "#### 42").
    """
    # Optional artificial latency to emulate high-latency tools for rollout
    # profiling. GSM8K_TOOL_SLEEP_MS is read per call so it can be tuned without
    # editing code. Runs in a worker thread (function tools are dispatched via
    # asyncio.to_thread), so it does not block other in-flight trajectories.
    sleep_ms = float(os.getenv("GSM8K_TOOL_SLEEP_MS", "0"))
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    parsed = _extract_number(answer)
    if parsed is None:
        return (
            "Could not parse a number from your answer. Make sure your final "
            "answer is in the format `#### <number>`, e.g. `#### 42`."
        )
    return f"Parsed answer: {parsed}. If you are confident, output it as `#### {parsed}`."
