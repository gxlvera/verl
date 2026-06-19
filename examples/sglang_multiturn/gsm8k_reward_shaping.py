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
"""GSM8K tool-agent reward with tool-call shaping.

Outcome reward = standard GSM8K rule-based correctness (1.0 correct / 0.0 wrong).

Tool-call shaping (avoids the "collapse to 2 turns / no tool call" failure mode
reported in verl issue #1569 and fixed by PR #4998): add a small positive bonus
when the trajectory actually used the ``calc_gsm8k_reward`` tool at least once.

In the tool_agent loop, a no-tool trajectory is ``[user, assistant]`` (2 turns),
while a tool-using trajectory is ``[user, assistant, tool, assistant, ...]``
(>= 4 turns). The ``naive`` reward manager injects ``__num_turns__`` into
``extra_info["num_turns"]``, so ``num_turns > 2`` is a reliable "used a tool"
signal. We additionally check for ``<tool_call>`` in the decoded response as a
fallback.

Wire it up via::

    reward.reward_manager.name=naive
    reward.custom_reward_function.path=examples/sglang_multiturn/gsm8k_reward_shaping.py
    reward.custom_reward_function.name=compute_score
"""

from __future__ import annotations

import os
from typing import Any, Optional

from verl.utils.reward_score import gsm8k

# Bonus added to the outcome reward when the trajectory used the tool at least
# once. Override with GSM8K_TOOL_SHAPING_BONUS. PR #4998 used 0.1.
_TOOL_SHAPING_BONUS = float(os.getenv("GSM8K_TOOL_SHAPING_BONUS", "0.1"))


def _used_tool(solution_str: str, num_turns: Optional[int]) -> bool:
    if num_turns is not None and num_turns > 2:
        return True
    # Fallback: the assistant emitted a tool call in its response stream.
    return "<tool_call>" in (solution_str or "")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs,
) -> dict[str, Any]:
    extra_info = extra_info or {}

    # 1) outcome reward: GSM8K rule-based correctness.
    acc = gsm8k.compute_score(
        solution_str,
        ground_truth,
        method="flexible",
        format_score=0.0,
        score=1.0,
    )
    acc = float(acc)

    # 2) tool-call shaping bonus.
    num_turns = extra_info.get("num_turns")
    used_tool = _used_tool(solution_str, num_turns)
    tool_bonus = _TOOL_SHAPING_BONUS if used_tool else 0.0

    score = acc + tool_bonus

    # Returning a dict makes the naive reward manager log every key into
    # reward_extra_info, so `acc`, `used_tool`, `num_turns` show up in the
    # training metrics / console for easy debugging.
    return {
        "score": score,
        "acc": acc,
        "used_tool": float(used_tool),
        "num_turns": float(num_turns) if num_turns is not None else 0.0,
    }
