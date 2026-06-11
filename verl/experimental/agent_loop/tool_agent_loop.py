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
import asyncio
import contextlib
import json
import logging
import os
import random
import re
import time
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        state_tracker: Any = None,
        state_sample_id: Optional[int] = None,
    ):
        self.messages = messages
        # Parallel message history for the COMPRESSED non-thinking shadow prompt:
        # system + user + (assistant tool_call WITHOUT thinking) + (tool response)
        # alternating, so the non-thinking shadow never sees the main model's prior
        # <think> traces. Maintained only when HOTPOT_SHADOW_COMPRESS is on.
        self.nonthinking_msgs: list[dict[str, Any]] = [dict(m) for m in messages]
        # Parallel accumulated token_ids for the COMPRESSED direct-token shadow:
        # mirrors prompt_ids but with each prior assistant turn's <think>...</think>
        # span removed at the token level. Maintained only when shadow_compress +
        # shadow_direct_tokens are on.
        self.shadow_prompt_ids: list[int] = []
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.state_tracker = state_tracker
        self.state_sample_id = state_sample_id

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Per-turn timing for fine-grained rollout profiling. One entry is
        # appended to per_turn_decode for every assistant generation, and one to
        # per_turn_tool for every turn that triggers a tool call.
        self.per_turn_decode: list[float] = []
        self.per_turn_tool: list[float] = []

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.tool_call_count = 0
        self.prefetched_tool_responses: dict[str, tuple[ToolResponse, float, dict]] = {}
        self.speculative_records: list[dict[str, Any]] = []
        self._speculative_background_tasks: list[asyncio.Task] = []
        # Simulated-speculation state. _spec_hit is decided once per sample (None
        # until the sample first fires a tail shadow): True => this sample acts on
        # the non-thinking shadow by launching its tool call early. _spec_tool_task
        # holds that early-launched tool coroutine so the processing-tools state can
        # reuse it instead of issuing a fresh (full-latency) tool call.
        self._spec_hit: Optional[bool] = None
        self._spec_tool_task: Optional[asyncio.Task] = None

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}

    def set_worker_state(self, state: str | None) -> None:
        if self.state_tracker is not None and self.state_sample_id is not None:
            self.state_tracker.transition(self.state_sample_id, state)


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    # Process-local long-tail awareness. Every active trajectory (running run())
    # in this worker process increments _active_count; it is decremented when the
    # trajectory finishes. _active_peak tracks the high-water mark so we can define
    # the "tail" relative to this worker's own chunk size. _shadow_fired counts how
    # many shadow requests this process has launched.
    _active_count: int = 0
    _active_peak: int = 0
    _shadow_fired: int = 0
    # Long-tail checkpoint state (process-local). _tail_fracs are the active-count
    # thresholds (as a fraction of _active_peak). _tail_marked_at[frac] is the
    # perf_counter when the tail first dropped to that level; _tail_tool_after[frac]
    # counts tool calls fired at/after that mark.
    _tail_fracs = (0.25, 0.15, 0.10, 0.05)
    _tail_marked_at: dict[float, float] = {}
    _tail_tool_after: dict[float, int] = {}
    _tail_total_tool: int = 0
    _tail_last_update: float = 0.0
    _tail_last_write: float = 0.0
    # Process-level RNG for the simulated speculation hit decision. Must be
    # class-level: a fresh ToolAgentLoop is instantiated per trajectory, so a
    # per-instance seeded RNG would reset every time and make the hit decision
    # deterministic (always the seed's first draw).
    _spec_rng: Optional[random.Random] = None

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop.
        """
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        self.min_tool_calls = int(
            os.getenv("HOTPOT_MIN_TOOL_CALLS", os.getenv("GSM8K_MIN_TOOL_CALLS", "0")) or "0"
        )
        # Hard cap on tool calls per trajectory (0 = unlimited). Once reached, the
        # trajectory stops issuing further tool calls and terminates.
        self.max_tool_calls = int(
            os.getenv("HOTPOT_MAX_TOOL_CALLS", os.getenv("GSM8K_MAX_TOOL_CALLS", "0")) or "0"
        )
        self.auto_tool_name = os.getenv(
            "HOTPOT_AUTO_TOOL_NAME", os.getenv("GSM8K_AUTO_TOOL_NAME", "calc_gsm8k_reward")
        )
        per_turn_limit = os.getenv("AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH", "").strip()
        self.per_turn_response_length = int(per_turn_limit) if per_turn_limit else None
        self.speculative_prefetch = os.getenv("HOTPOT_SPECULATIVE_TOOL_PREFETCH", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self.speculative_jsonl = os.getenv("HOTPOT_SPECULATIVE_JSONL", "").strip()
        # When set, append one JSON record per assistant generation turn capturing
        # the SGLang waiting-queue time (extra_fields["sglang_queue_time_s"]) so we
        # can study, e.g., how the 2nd-turn (first tool-call return) requests queue
        # before their prefill. A per-pid suffix avoids multi-process write races.
        self.queue_time_jsonl = os.getenv("AGENT_LOOP_QUEUE_TIME_JSONL", "").strip()
        self._queue_time_fh = None
        if self.queue_time_jsonl:
            try:
                path = f"{self.queue_time_jsonl}.{os.getpid()}.jsonl"
                self._queue_time_fh = open(path, "a", buffering=1)
            except Exception:
                self._queue_time_fh = None
        self.main_enable_thinking = os.getenv(
            "HOTPOT_MAIN_ENABLE_THINKING", "true" if self.speculative_prefetch else ""
        ).strip()
        self.non_thinking_max_new_tokens = int(os.getenv("HOTPOT_NON_THINKING_MAX_NEW_TOKENS", "0") or "0")
        # Simple "shadow" load: when set, every assistant generation turn fires an
        # extra parallel enable_thinking=False request for the same sample. Its
        # output is discarded -- no tool-call matching / prefetch / reuse. It only
        # exists to double the concurrent request load so we can measure the effect
        # on rollout throughput and SGLang scheduling.
        self.shadow_nonthinking = os.getenv("HOTPOT_SHADOW_NONTHINKING", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        # Tail-only shadow: instead of always shadowing (above), only fire an extra
        # enable_thinking=False shadow request for a sample once (a) it has reached a
        # late turn (turn >= shadow_tail_min_turn) AND (b) this worker process is in
        # its long tail -- i.e. the number of still-running trajectories has dropped
        # to/below a threshold, which means SGLang load + KV are low and there is
        # spare decode bandwidth to soak up. This targets only the stragglers.
        self.shadow_tail = os.getenv("HOTPOT_SHADOW_TAIL", "").strip().lower() in {"1", "true", "yes"}
        # Compressed shadow prompt: feed the non-thinking shadow a context that has
        # the prior <think> traces stripped (only assistant tool_calls + tool
        # responses). Tests whether the main model's thinking interferes with the
        # non-thinking shadow and makes it ramble in later turns.
        self.shadow_compress = os.getenv("HOTPOT_SHADOW_COMPRESS", "").strip().lower() in {"1", "true", "yes"}
        # Direct-token shadow prompt: instead of re-running apply_chat_template (which
        # left-truncates to rollout.prompt_length and drops the system prompt/question
        # in later turns), feed the shadow the SAME accumulated token_ids as the main
        # thinking request, then append the empty-think prefix "<think>\n\n</think>\n\n"
        # to switch the model into non-thinking mode. No re-template, no truncation.
        self.shadow_direct_tokens = os.getenv("HOTPOT_SHADOW_DIRECT_TOKENS", "").strip().lower() in {"1", "true", "yes"}
        self._empty_think_suffix_ids: Optional[list[int]] = None
        self._think_open_id: Optional[int] = None
        self._think_close_id: Optional[int] = None
        self._think_ids_ready = False
        # Long-tail checkpoint stats: when the number of still-active trajectories in
        # this worker drops to <=25/15/10/5% of the per-worker peak, stamp a time and
        # then count how many tool calls fire AFTER that point (i.e. within the tail
        # window only). Written per-pid to HOTPOT_TAIL_CKPT.<pid>.json.
        self.tail_ckpt_path = os.getenv("HOTPOT_TAIL_CKPT", "").strip()
        self.shadow_tail_min_turn = int(os.getenv("HOTPOT_SHADOW_TAIL_MIN_TURN", "3") or "3")
        # Absolute active-trajectory ceiling for "in tail"; if unset/0, fall back to
        # a fraction of the per-worker peak active count.
        self.shadow_tail_active_max = int(os.getenv("HOTPOT_SHADOW_TAIL_ACTIVE_MAX", "0") or "0")
        self.shadow_tail_active_frac = float(os.getenv("HOTPOT_SHADOW_TAIL_ACTIVE_FRAC", "0.25") or "0.25")
        self.shadow_jsonl = os.getenv("HOTPOT_SHADOW_JSONL", "").strip()
        self._shadow_fh = None
        if self.shadow_jsonl:
            try:
                self._shadow_fh = open(f"{self.shadow_jsonl}.{os.getpid()}.jsonl", "a", buffering=1)
            except Exception:
                self._shadow_fh = None
        # Optional trace dump: write the FULL prompt (decoded text) fed to the LLM at
        # selected turns, so one can inspect the accumulated multi-turn context. Gated
        # by HOTPOT_TRACE_DUMP (file prefix); turns via HOTPOT_TRACE_DUMP_TURNS
        # ("3,4"), capped by HOTPOT_TRACE_DUMP_MAX per worker (default 30).
        self.trace_dump_path = os.getenv("HOTPOT_TRACE_DUMP", "").strip()
        # Turns to dump. Special value "all" (or "0") => dump EVERY turn from turn 1.
        _turns_env = os.getenv("HOTPOT_TRACE_DUMP_TURNS", "4").strip().lower()
        if _turns_env in ("all", "0", "*"):
            self.trace_dump_turns = None  # None => all turns
        else:
            self.trace_dump_turns = {
                int(x) for x in _turns_env.split(",") if x.strip()
            }
        self.trace_dump_max = int(os.getenv("HOTPOT_TRACE_DUMP_MAX", "30") or "30")
        self._trace_fh = None
        self._trace_dumped = 0
        if self.trace_dump_path:
            try:
                self._trace_fh = open(f"{self.trace_dump_path}.{os.getpid()}.jsonl", "a", buffering=1)
            except Exception:
                self._trace_fh = None
        # Simulated speculation hit rate: fraction of tail-shadow SAMPLES that, on
        # receiving the non-thinking shadow response, immediately fire the tool call
        # (overlapping tool latency with the main thinking decode) and then skip the
        # real tool call when the main response arrives. The rest discard the shadow.
        self.spec_hit_rate = float(os.getenv("HOTPOT_SPEC_HIT_RATE", "0") or "0")
        # Optional per-turn hit rates, e.g. "0.30,0.30,0.18,0.15" -> turn1..turn4
        # (turns beyond the list reuse the last value). When set, the hit/miss is
        # rolled PER TURN (every turn a shadow fires) using that turn's rate,
        # instead of once per sample. Used for full-shadow where early thinking
        # turns are more speculatable than later ones.
        _by_turn = os.getenv("HOTPOT_SPEC_HIT_RATE_BY_TURN", "").strip()
        self.spec_hit_by_turn = [float(x) for x in _by_turn.split(",") if x.strip()] if _by_turn else []
        # When true (default) the shadow is fire-and-forget: the main path is never
        # blocked on the shadow and the shadow is cancelled once the main returns.
        # Set false to reproduce the old behaviour where the turn awaits BOTH the
        # main and shadow (asyncio.gather), which puts the shadow on the critical path.
        self.shadow_fire_and_forget = os.getenv("HOTPOT_SHADOW_FIRE_AND_FORGET", "true").strip().lower() in ("1", "true", "yes")
        if ToolAgentLoop._spec_rng is None:
            _spec_seed = os.getenv("HOTPOT_SPEC_HIT_SEED", "").strip()
            ToolAgentLoop._spec_rng = random.Random(int(_spec_seed)) if _spec_seed else random.Random()

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # Effective model context limit, used to gracefully truncate (terminate)
        # over-long multi-turn trajectories instead of letting SGLang reject them.
        self.max_model_len = self.rollout_config.max_model_len or (self.prompt_length + self.response_length)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        _agent_loop_start = time.perf_counter()
        messages = list(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        state_tracker = kwargs.get("_agent_state_tracker")
        state_sample_id = kwargs.get("_agent_state_sample_id")

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            state_tracker=state_tracker,
            state_sample_id=state_sample_id,
        )

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING
        ToolAgentLoop._active_count += 1
        ToolAgentLoop._active_peak = max(ToolAgentLoop._active_peak, ToolAgentLoop._active_count)
        try:
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.PROCESSING_TOOLS:
                    state = await self._handle_processing_tools_state(agent_data)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED
        finally:
            ToolAgentLoop._active_count -= 1
            self._maybe_mark_tail_checkpoint()
            agent_data.set_worker_state("finished")
            for task in agent_data._speculative_background_tasks:
                if not task.done():
                    task.cancel()
            for task in agent_data._speculative_background_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        agent_data.metrics["agent_loop_e2e"] = time.perf_counter() - _agent_loop_start

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        # Per-turn total = each turn's decode plus the tool call that follows it
        # (the final turn typically has no tool call, contributing decode only).
        agent_data.metrics["per_turn_decode"] = agent_data.per_turn_decode
        agent_data.metrics["per_turn_tool"] = agent_data.per_turn_tool
        agent_data.metrics["per_turn_total"] = [
            decode + (agent_data.per_turn_tool[i] if i < len(agent_data.per_turn_tool) else 0.0)
            for i, decode in enumerate(agent_data.per_turn_decode)
        ]
        self._finalize_tool_latency_metrics(agent_data)
        self._finalize_speculative_metrics(agent_data)
        self._write_speculative_jsonl(agent_data)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            apply_chat_template_kwargs=self._main_chat_template_kwargs(),
        )
        agent_data.prompt_ids = prompt_ids
        if self.shadow_compress and self.shadow_direct_tokens:
            agent_data.shadow_prompt_ids = list(prompt_ids)
        return AgentState.GENERATING

    def _main_chat_template_kwargs(self) -> dict[str, Any] | None:
        if not self.main_enable_thinking:
            return None
        return {"enable_thinking": self.main_enable_thinking.lower() in {"1", "true", "yes"}}

    def _non_thinking_sampling_params(self, sampling_params: dict[str, Any]) -> dict[str, Any]:
        params = dict(sampling_params)
        if self.non_thinking_max_new_tokens > 0:
            params["max_new_tokens"] = min(
                self.non_thinking_max_new_tokens,
                int(params.get("max_new_tokens", self.non_thinking_max_new_tokens)),
            )
        return params

    def _should_shadow_tail(self, turn_index: int) -> bool:
        """Tail-only shadow gate (VERL-side long-tail awareness).

        Returns True iff tail-shadow mode is on, this generation is at a late turn
        (>= shadow_tail_min_turn), and the worker process is currently in its long
        tail -- i.e. few trajectories remain active, which implies SGLang load and
        KV occupancy are low and there is spare decode bandwidth.
        """
        if not self.shadow_tail:
            return False
        if turn_index < self.shadow_tail_min_turn:
            return False
        if self.shadow_tail_active_max > 0:
            ceiling = self.shadow_tail_active_max
        else:
            ceiling = max(1, int(self.shadow_tail_active_frac * ToolAgentLoop._active_peak))
        return ToolAgentLoop._active_count <= ceiling

    def _maybe_mark_tail_checkpoint(self) -> None:
        """Called when a trajectory finishes (active_count just decremented). The
        first time the active count drops to <= frac*peak, stamp the time so we can
        later attribute tool calls to the tail window."""
        if not self.tail_ckpt_path:
            return
        peak = ToolAgentLoop._active_peak
        if peak <= 0:
            return
        marked = False
        for frac in ToolAgentLoop._tail_fracs:
            if frac in ToolAgentLoop._tail_marked_at:
                continue
            if ToolAgentLoop._active_count <= frac * peak:
                ToolAgentLoop._tail_marked_at[frac] = time.perf_counter()
                ToolAgentLoop._tail_tool_after.setdefault(frac, 0)
                marked = True
        # Write on a new mark, and force a final flush when this worker drains
        # (last trajectory finished) so throttled tail counts are not lost.
        if marked or ToolAgentLoop._active_count <= 0:
            self._write_tail_ckpt()

    def _count_tail_toolcalls(self, n: int) -> None:
        """Count n tool calls against every tail checkpoint already crossed."""
        if not self.tail_ckpt_path or n <= 0:
            return
        ToolAgentLoop._tail_total_tool += n
        if ToolAgentLoop._tail_marked_at:
            for frac in ToolAgentLoop._tail_marked_at:
                ToolAgentLoop._tail_tool_after[frac] = ToolAgentLoop._tail_tool_after.get(frac, 0) + n
            # Throttle: snapshot at most once per second so write frequency is
            # independent of tool-call volume (keeps it off the latency path).
            if time.perf_counter() - ToolAgentLoop._tail_last_write > 1.0:
                self._write_tail_ckpt()

    def _write_tail_ckpt(self) -> None:
        ToolAgentLoop._tail_last_update = time.perf_counter()
        ToolAgentLoop._tail_last_write = ToolAgentLoop._tail_last_update
        try:
            snap = {
                "pid": os.getpid(),
                "active_peak": ToolAgentLoop._active_peak,
                "total_tool_calls": ToolAgentLoop._tail_total_tool,
                "last_update": ToolAgentLoop._tail_last_update,
                "checkpoints": {
                    str(frac): {
                        "marked_at": ToolAgentLoop._tail_marked_at[frac],
                        "tool_calls_after": ToolAgentLoop._tail_tool_after.get(frac, 0),
                        "window_s": ToolAgentLoop._tail_last_update - ToolAgentLoop._tail_marked_at[frac],
                    }
                    for frac in ToolAgentLoop._tail_marked_at
                },
            }
            with open(f"{self.tail_ckpt_path}.{os.getpid()}.json", "w") as fh:
                json.dump(snap, fh)
        except Exception:
            pass

    def _log_shadow_fire(self, agent_data: AgentData, turn_index: int) -> None:
        ToolAgentLoop._shadow_fired += 1
        if self._shadow_fh is None:
            return
        try:
            self._shadow_fh.write(
                json.dumps(
                    {
                        "request_id": agent_data.request_id,
                        "turn": int(turn_index),
                        "active_at_fire": ToolAgentLoop._active_count,
                        "active_peak": ToolAgentLoop._active_peak,
                        "prompt_len": len(agent_data.prompt_ids),
                    }
                )
                + "\n"
            )
        except Exception:
            pass

    def _log_shadow_decode(
        self,
        agent_data: AgentData,
        turn_index: int,
        decode_s: float,
        hit: bool,
        main: bool = False,
        tokens: Optional[int] = None,
        queue_s: Optional[float] = None,
    ) -> None:
        if self._shadow_fh is None:
            return
        try:
            self._shadow_fh.write(
                json.dumps(
                    {
                        "event": "main_decode" if main else "shadow_decode",
                        "request_id": agent_data.request_id,
                        "turn": int(turn_index),
                        "decode_s": float(decode_s),
                        "tokens": int(tokens) if tokens is not None else None,
                        "queue_s": float(queue_s) if queue_s is not None else None,
                        "spec_hit_sample": bool(hit),
                    }
                )
                + "\n"
            )
        except Exception:
            pass

    def _maybe_dump_trace(self, agent_data: AgentData, turn_index: int) -> None:
        """Dump the full decoded prompt fed to the LLM at selected turns."""
        if self._trace_fh is None or (self.trace_dump_turns is not None and turn_index not in self.trace_dump_turns):
            return
        if self._trace_dumped >= self.trace_dump_max:
            return
        try:
            text = self.tokenizer.decode(agent_data.prompt_ids, skip_special_tokens=False)
            self._trace_fh.write(
                json.dumps(
                    {
                        "request_id": agent_data.request_id,
                        "turn": turn_index,
                        "prompt_len": len(agent_data.prompt_ids),
                        "prompt_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._trace_dumped += 1
        except Exception:
            pass

    def _maybe_dump_shadow_text(self, agent_data: AgentData, turn_index: int, out: "TokenOutput") -> None:
        """Dump the decoded non-thinking shadow OUTPUT text at selected turns."""
        if self._trace_fh is None or (self.trace_dump_turns is not None and turn_index not in self.trace_dump_turns):
            return
        if self._trace_dumped >= self.trace_dump_max:
            return
        try:
            text = self.tokenizer.decode(out.token_ids, skip_special_tokens=False)
            self._trace_fh.write(
                json.dumps(
                    {
                        "kind": "shadow_output",
                        "request_id": agent_data.request_id,
                        "turn": turn_index,
                        "tokens": len(out.token_ids),
                        "shadow_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._trace_dumped += 1
        except Exception:
            pass

    def _maybe_dump_main_text(self, agent_data: AgentData, turn_index: int, out: "TokenOutput") -> None:
        """Dump the decoded main (thinking) OUTPUT text at selected turns, keyed by
        the same request_id+turn as the shadow dump so the two can be paired."""
        if self._trace_fh is None or (self.trace_dump_turns is not None and turn_index not in self.trace_dump_turns):
            return
        if self._trace_dumped >= self.trace_dump_max:
            return
        try:
            text = self.tokenizer.decode(out.token_ids, skip_special_tokens=False)
            self._trace_fh.write(
                json.dumps(
                    {
                        "kind": "main_output",
                        "request_id": agent_data.request_id,
                        "turn": turn_index,
                        "tokens": len(out.token_ids),
                        "main_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._trace_dumped += 1
        except Exception:
            pass

    def _maybe_dump_shadow_prompt(self, agent_data: AgentData, turn_index: int, shadow_prompt_ids: list[int]) -> None:
        """Dump the full decoded prompt fed to the NON-THINKING shadow at selected
        turns. With shadow compression on, this differs from the main prompt, so we
        record it separately (kind=shadow_prompt)."""
        if self._trace_fh is None or (self.trace_dump_turns is not None and turn_index not in self.trace_dump_turns):
            return
        if self._trace_dumped >= self.trace_dump_max:
            return
        try:
            text = self.tokenizer.decode(shadow_prompt_ids, skip_special_tokens=False)
            self._trace_fh.write(
                json.dumps(
                    {
                        "kind": "shadow_prompt",
                        "request_id": agent_data.request_id,
                        "turn": turn_index,
                        "prompt_len": len(shadow_prompt_ids),
                        "shadow_prompt_text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._trace_dumped += 1
        except Exception:
            pass

    async def _generate_with_shadow(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> "TokenOutput":
        """Fire the main (thinking) generation plus one parallel enable_thinking=False
        shadow request, but FIRE-AND-FORGET the shadow: return as soon as the main
        finishes and cancel the shadow if it is still running. This keeps the shadow
        off the critical path so it cannot extend the makespan of tail stragglers.

        On a simulated-hit sample, if the (faster) shadow happens to finish before the
        main, it parses its tool call and launches the tool early so its latency can
        overlap the remaining main decode (processing-tools then reuses it).
        """
        turn_index = agent_data.assistant_turns + 1
        self._log_shadow_fire(agent_data, turn_index)
        # Decide hit/miss. With per-turn rates (HOTPOT_SPEC_HIT_RATE_BY_TURN) the
        # roll happens EVERY turn a shadow fires, using that turn's rate. Otherwise
        # fall back to a single uniform rate (HOTPOT_SPEC_HIT_RATE), decided once
        # per sample so a sample consistently hits/misses.
        if self.spec_hit_by_turn:
            rate = self.spec_hit_by_turn[min(turn_index, len(self.spec_hit_by_turn)) - 1]
            agent_data._spec_hit = ToolAgentLoop._spec_rng.random() < rate
        elif agent_data._spec_hit is None and self.spec_hit_rate > 0:
            agent_data._spec_hit = ToolAgentLoop._spec_rng.random() < self.spec_hit_rate

        t_fire = time.perf_counter()
        main_task = asyncio.create_task(
            self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        )
        shadow_prompt_ids = await self._build_non_thinking_prompt_ids(agent_data)
        self._maybe_dump_shadow_prompt(agent_data, turn_index, shadow_prompt_ids)

        async def _run_shadow() -> None:
            out = await self.server_manager.generate(
                request_id=f"{agent_data.request_id}-shadow-{turn_index}",
                prompt_ids=shadow_prompt_ids,
                sampling_params=self._non_thinking_sampling_params(sampling_params),
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
            self._log_shadow_decode(
                agent_data,
                turn_index,
                time.perf_counter() - t_fire,
                hit=bool(agent_data._spec_hit),
                tokens=len(out.token_ids),
                queue_s=(out.extra_fields or {}).get("sglang_queue_time_s"),
            )
            self._maybe_dump_shadow_text(agent_data, turn_index, out)
            # Simulated hit: the shadow beat the main, so launch its tool now to
            # overlap the tool latency with the remaining main decode.
            if agent_data._spec_hit and agent_data._spec_tool_task is None:
                calls = await self._parse_shadow_tool_calls(agent_data, out)
                if calls:
                    agent_data._spec_tool_task = asyncio.create_task(
                        self._call_tool(calls[0], agent_data.tools_kwargs, agent_data)
                    )
                    agent_data._speculative_background_tasks.append(agent_data._spec_tool_task)

        shadow_task = asyncio.create_task(_run_shadow())
        agent_data._speculative_background_tasks.append(shadow_task)

        output = await main_task
        main_decode_s = time.perf_counter() - t_fire
        if self.shadow_fire_and_forget:
            # Fire-and-forget: do NOT wait for the shadow. Cancel it if it has not
            # finished by the time the main returns so it never extends the turn.
            if not shadow_task.done():
                shadow_task.cancel()
        else:
            # Legacy behaviour: block the turn until the shadow also finishes,
            # putting the (often slower) shadow on the critical path.
            try:
                await shadow_task
            except asyncio.CancelledError:
                pass
        self._log_shadow_decode(
            agent_data,
            turn_index,
            main_decode_s,
            hit=bool(agent_data._spec_hit),
            main=True,
            tokens=len(output.token_ids),
            queue_s=(output.extra_fields or {}).get("sglang_queue_time_s"),
        )
        self._maybe_dump_main_text(agent_data, turn_index, output)
        return output

    async def _parse_shadow_tool_calls(self, agent_data: AgentData, shadow_out: "TokenOutput") -> list[FunctionCall]:
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        try:
            _, calls = await self.tool_parser.extract_tool_calls(shadow_out.token_ids, tools)
        except Exception:
            calls = []
        return calls

    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    async def _append_compressed_assistant_turn(self, agent_data: AgentData) -> None:
        """Append the just-completed assistant turn to nonthinking_msgs with its
        <think>...</think> block stripped, so the compressed shadow context keeps
        only the assistant's tool_call (no reasoning)."""
        if not self.shadow_compress or self.shadow_direct_tokens:
            return
        text = await self._decode_current_response(agent_data.response_ids)
        text = self._THINK_RE.sub("", text).strip()
        agent_data.nonthinking_msgs.append({"role": "assistant", "content": text})

    def _ensure_think_ids(self) -> None:
        """Lazily resolve the <think>/</think> token ids and the empty-think prefix
        token ids used to switch the model into non-thinking mode."""
        if self._think_ids_ready:
            return
        self._think_ids_ready = True
        try:
            self._empty_think_suffix_ids = self.tokenizer.encode(
                "<think>\n\n</think>\n\n", add_special_tokens=False
            )
        except Exception:
            self._empty_think_suffix_ids = []
        for tok, attr in (("<think>", "_think_open_id"), ("</think>", "_think_close_id")):
            try:
                ids = self.tokenizer.encode(tok, add_special_tokens=False)
                setattr(self, attr, ids[0] if len(ids) == 1 else None)
            except Exception:
                setattr(self, attr, None)

    def _strip_think_tokens(self, ids: list[int]) -> list[int]:
        """Remove the FIRST <think>...</think> span (inclusive) from a token list.
        Used on each assistant turn's generated tokens, where any <think>/</think>
        come from the model's own reasoning (never the system prompt)."""
        self._ensure_think_ids()
        o, c = self._think_open_id, self._think_close_id
        if o is None or c is None:
            return list(ids)
        try:
            i = ids.index(o)
            j = ids.index(c, i)
        except ValueError:
            return list(ids)
        return ids[:i] + ids[j + 1 :]

    def _direct_shadow_prompt_ids(self, agent_data: AgentData) -> list[int]:
        """Build the shadow prompt by reusing the main's accumulated token_ids
        (no re-template, no truncation) and appending the empty-think prefix so the
        model generates in non-thinking mode. In compress mode the base is the
        think-stripped history (agent_data.shadow_prompt_ids)."""
        self._ensure_think_ids()
        base = agent_data.shadow_prompt_ids if self.shadow_compress else agent_data.prompt_ids
        return list(base) + list(self._empty_think_suffix_ids or [])

    async def _build_non_thinking_prompt_ids(self, agent_data: AgentData) -> list[int]:
        if self.shadow_direct_tokens:
            return self._direct_shadow_prompt_ids(agent_data)
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        msgs = agent_data.nonthinking_msgs if self.shadow_compress else agent_data.messages
        return await self.apply_chat_template(
            msgs,
            tools=schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
            audios=agent_data.audio_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            apply_chat_template_kwargs={"enable_thinking": False},
        )

    @staticmethod
    def _tool_call_key(tool_call: FunctionCall) -> str:
        try:
            args = json.loads(tool_call.arguments)
            args_text = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except Exception:
            args_text = str(tool_call.arguments)
        return json.dumps({"name": tool_call.name, "arguments": args_text}, sort_keys=True, ensure_ascii=False)

    def _tool_calls_match(self, main_calls: list[FunctionCall], draft_calls: list[FunctionCall]) -> bool:
        if len(main_calls) != len(draft_calls):
            return False
        return [self._tool_call_key(c) for c in main_calls] == [self._tool_call_key(c) for c in draft_calls]

    async def _generate_main_and_speculative(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
    ) -> tuple[TokenOutput, list[FunctionCall], dict[str, Any] | None]:
        turn_index = agent_data.assistant_turns + 1
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        record: dict[str, Any] = {
            "turn": turn_index,
            "sample_request_id": agent_data.request_id,
            "enabled": True,
            "main_submit_epoch_s": time.time(),
            "main_submit_perf_s": time.perf_counter(),
        }
        non_prompt_ids = await self._build_non_thinking_prompt_ids(agent_data)
        record["non_thinking_submit_epoch_s"] = time.time()
        record["non_thinking_submit_perf_s"] = time.perf_counter()

        main_task = asyncio.create_task(
            self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        )
        draft_task = asyncio.create_task(
            self.server_manager.generate(
                request_id=f"{agent_data.request_id}-nonthinking-{turn_index}",
                prompt_ids=non_prompt_ids,
                sampling_params=self._non_thinking_sampling_params(sampling_params),
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        )

        draft_calls: list[FunctionCall] = []
        prefetch_task: asyncio.Task | None = None
        prefetch_call: FunctionCall | None = None

        while not main_task.done():
            done, _ = await asyncio.wait({main_task, draft_task}, return_when=asyncio.FIRST_COMPLETED)
            if draft_task in done and not record.get("non_thinking_done_epoch_s"):
                record["non_thinking_done_epoch_s"] = time.time()
                record["non_thinking_done_perf_s"] = time.perf_counter()
                draft_output = draft_task.result()
                record["non_thinking_tokens"] = len(draft_output.token_ids)
                _, draft_calls = await self.tool_parser.extract_tool_calls(draft_output.token_ids, tools)
                record["non_thinking_tool_calls"] = [c.model_dump() for c in draft_calls]
                if draft_calls:
                    prefetch_call = draft_calls[0]
                    record["prefetch_tool_submit_epoch_s"] = time.time()
                    record["prefetch_tool_submit_perf_s"] = time.perf_counter()
                    prefetch_task = asyncio.create_task(self._call_tool(prefetch_call, agent_data.tools_kwargs, agent_data))
                    agent_data._speculative_background_tasks.append(prefetch_task)
            if main_task in done:
                break

        output = await main_task
        record["main_done_epoch_s"] = time.time()
        record["main_done_perf_s"] = time.perf_counter()
        record["main_tokens"] = len(output.token_ids)
        if not draft_task.done():
            draft_output = await draft_task
            record["non_thinking_done_epoch_s"] = time.time()
            record["non_thinking_done_perf_s"] = time.perf_counter()
            record["non_thinking_tokens"] = len(draft_output.token_ids)
            _, draft_calls = await self.tool_parser.extract_tool_calls(draft_output.token_ids, tools)
            record["non_thinking_tool_calls"] = [c.model_dump() for c in draft_calls]
            if draft_calls and prefetch_task is None:
                prefetch_call = draft_calls[0]
                record["prefetch_tool_submit_epoch_s"] = time.time()
                record["prefetch_tool_submit_perf_s"] = time.perf_counter()
                prefetch_task = asyncio.create_task(self._call_tool(prefetch_call, agent_data.tools_kwargs, agent_data))
                agent_data._speculative_background_tasks.append(prefetch_task)

        _, main_calls = await self.tool_parser.extract_tool_calls(output.token_ids, tools)
        record["main_tool_calls"] = [c.model_dump() for c in main_calls]
        record["main_finished"] = not bool(main_calls)
        match = bool(main_calls) and self._tool_calls_match(main_calls, draft_calls)
        record["tool_call_match"] = match

        if prefetch_task is not None:
            if match:
                result = await prefetch_task
                record["prefetch_tool_done_epoch_s"] = time.time()
                record["prefetch_tool_done_perf_s"] = time.perf_counter()
                record["prefetch_delta_after_main_s"] = (
                    record["prefetch_tool_done_perf_s"] - record["main_done_perf_s"]
                )
                record["prefetch_reused"] = True
                if prefetch_call is not None:
                    agent_data.prefetched_tool_responses[self._tool_call_key(prefetch_call)] = result
                    self._record_tool_result(agent_data, result, source="prefetch_reused")
            elif prefetch_task.done():
                with contextlib.suppress(Exception):
                    result = prefetch_task.result()
                    record["prefetch_tool_done_epoch_s"] = time.time()
                    record["prefetch_tool_done_perf_s"] = time.perf_counter()
                    self._record_tool_result(agent_data, result, source="prefetch_wasted")
                record["prefetch_reused"] = False
            else:
                prefetch_task.cancel()
                record["prefetch_cancelled"] = True
                record["prefetch_reused"] = False

        agent_data.speculative_records.append(record)
        return output, main_calls, record

    def _record_queue_time(self, agent_data: AgentData, output: "TokenOutput", turn_index: int) -> None:
        """Append the SGLang waiting-queue time for the generation that just ran.

        No-op unless AGENT_LOOP_QUEUE_TIME_JSONL is set. queue_time_s comes from
        SGLang meta_info["queue_time"] (forward_entry_time - wait_queue_entry_time).
        """
        if self._queue_time_fh is None:
            return
        try:
            qt = (output.extra_fields or {}).get("sglang_queue_time_s")
            record = {
                "request_id": agent_data.request_id,
                "turn": int(turn_index),
                "queue_time_s": float(qt) if qt is not None else None,
                "prompt_len": len(agent_data.prompt_ids),
                "response_tokens": len(output.token_ids),
                "global_steps": (output.extra_fields or {}).get("global_steps"),
            }
            self._queue_time_fh.write(json.dumps(record) + "\n")
        except Exception:
            pass

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # Inject tool parser stop tokens so generation halts after each tool call
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}
        # Graceful truncation: if the accumulated context is already at/over the
        # model's context window, stop the trajectory instead of sending an
        # over-length request that SGLang would reject with a ValueError.
        context_budget = self.max_model_len - len(agent_data.prompt_ids) - 1
        if context_budget <= 0:
            return AgentState.TERMINATED
        if self.per_turn_response_length is not None:
            remaining_response = self.response_length - len(agent_data.response_mask)
            if remaining_response <= 0:
                return AgentState.TERMINATED
            sampling_params = {
                **sampling_params,
                "max_new_tokens": min(self.per_turn_response_length, remaining_response, context_budget),
            }
        else:
            # Even without a per-turn cap, never request more than the remaining
            # context budget so multi-turn trajectories cannot overflow.
            sampling_params = {
                **sampling_params,
                "max_new_tokens": min(self.response_length, context_budget),
            }

        _decode_start = time.perf_counter()
        agent_data.set_worker_state("ready_to_llm")
        with simple_timer("generate_sequences", agent_data.metrics):
            agent_data.set_worker_state("llm_inflight")
            turn_index = agent_data.assistant_turns + 1
            self._maybe_dump_trace(agent_data, turn_index)
            if self.speculative_prefetch:
                output, parsed_tool_calls, _spec_record = await self._generate_main_and_speculative(
                    agent_data, sampling_params
                )
            elif self.shadow_nonthinking or self._should_shadow_tail(turn_index):
                output = await self._generate_with_shadow(agent_data, sampling_params)
                parsed_tool_calls = None
            else:
                output: TokenOutput = await self.server_manager.generate(
                    request_id=agent_data.request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                    audio_data=agent_data.audio_data,
                    mm_processor_kwargs=agent_data.mm_processor_kwargs,
                )
                parsed_tool_calls = None
        agent_data.set_worker_state(None)
        agent_data.per_turn_decode.append(time.perf_counter() - _decode_start)
        # Turn index of the generation that just completed (1-based): turn 1 is the
        # initial question, turn 2 is the first tool-call return, etc.
        self._record_queue_time(agent_data, output, turn_index=agent_data.assistant_turns + 1)
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        if self.shadow_compress and self.shadow_direct_tokens:
            # Append this assistant turn to the compressed shadow history with its
            # <think>...</think> reasoning span stripped at the token level.
            agent_data.shadow_prompt_ids += self._strip_think_tokens(agent_data.response_ids)
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        if parsed_tool_calls is None:
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        else:
            agent_data.tool_calls = parsed_tool_calls

        # Hard cap on tool calls: once reached, ignore further tool calls and stop.
        if self.max_tool_calls and agent_data.tool_call_count >= self.max_tool_calls:
            return AgentState.TERMINATED
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        if agent_data.tool_call_count < self.min_tool_calls and self.auto_tool_name in active_tools:
            answer_text = await self._decode_current_response(agent_data.response_ids)
            agent_data.tool_calls = [
                FunctionCall(name=self.auto_tool_name, arguments=json.dumps({"answer": answer_text}, ensure_ascii=False))
            ]
            return AgentState.PROCESSING_TOOLS
        else:
            return AgentState.TERMINATED

    async def _decode_current_response(self, response_ids: list[int]) -> str:
        return await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        # Record the assistant tool_call (without thinking) into the compressed
        # shadow history before appending this turn's tool responses below.
        await self._append_compressed_assistant_turn(agent_data)
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        responses: list[tuple[ToolResponse, float, dict]] = []
        tool_call_names = []
        # Simulated speculation hit: a tool task was already launched the moment the
        # non-thinking shadow returned, so its latency overlapped with the main
        # decode. Reuse it for the first tool call (await whatever time remains)
        # instead of issuing a fresh full-latency call.
        spec_tool_task = agent_data._spec_tool_task
        agent_data._spec_tool_task = None
        for idx, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
            key = self._tool_call_key(tool_call)
            if idx == 0 and spec_tool_task is not None:
                tasks.append(spec_tool_task)
            elif key in agent_data.prefetched_tool_responses:
                responses.append(agent_data.prefetched_tool_responses.pop(key))
            else:
                tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)
        _n_tool_this_turn = len(agent_data.tool_calls[: self.max_parallel_calls])
        agent_data.tool_call_count += _n_tool_this_turn
        agent_data.metrics["tool_call_count"] = agent_data.tool_call_count
        self._count_tail_toolcalls(_n_tool_this_turn)

        _tool_start = time.perf_counter()
        agent_data.set_worker_state("waiting_tool")
        with simple_timer("tool_calls", agent_data.metrics):
            if tasks:
                task_responses = await asyncio.gather(*tasks)
                for response in task_responses:
                    self._record_tool_result(agent_data, response, source="fallback_or_regular")
                responses.extend(task_responses)
        agent_data.set_worker_state(None)
        agent_data.per_turn_tool.append(time.perf_counter() - _tool_start)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        if self.shadow_compress:
            agent_data.nonthinking_msgs.extend(dict(m) for m in add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "gemma4":
            # Gemma4's chat template drops tool responses when passed without the preceding
            # assistant tool_call message. Manually format the response tokens.
            # Format: <|tool_response>response:func_name{value:<|"|>content<|"|>}<tool_response|>
            parts = []
            for msg, name in zip(add_messages, tool_call_names, strict=True):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
            tool_response_text = "".join(parts)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        if self.shadow_compress and self.shadow_direct_tokens:
            agent_data.shadow_prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle.
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        # Validate tool name
        tool_name = tool_call.name
        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Validate tool arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Execute tool
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name]

            if isinstance(tool, FunctionTool):
                # Function-based tools have no lifecycle. Dataset-provided
                # create_kwargs are injected as hidden runtime parameters, so they
                # are not exposed in the OpenAI tool schema seen by the model.
                kwargs = tools_kwargs.get(tool_name, {})
                raw = await tool.call(tool_args, injected_parameters=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
            else:
                # BaseTool subclass
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, res = await tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0, {}
        finally:
            # Only BaseTool instances need release (function tools never set instance_id).
            if tool and instance_id and not isinstance(tool, FunctionTool):
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            elif self.tool_response_truncate_side == "right":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _record_tool_result(
        self, agent_data: AgentData, result: tuple[ToolResponse, float, dict], *, source: str
    ) -> None:
        _tool_response, _reward, meta = result
        if not isinstance(meta, dict):
            return
        latency = meta.get("latency_s", meta.get("configured_sleep_s"))
        configured = meta.get("configured_sleep_s")
        if latency is not None:
            agent_data.metrics.setdefault("tool_latency_s", []).append(float(latency))
        if configured is not None:
            agent_data.metrics.setdefault("tool_configured_sleep_s", []).append(float(configured))
        agent_data.extra_fields.setdefault("tool_latency_source", []).append(source)

    def _finalize_tool_latency_metrics(self, agent_data: AgentData) -> None:
        for key in ("tool_latency_s", "tool_configured_sleep_s"):
            values = agent_data.metrics.get(key) or []
            if not values:
                continue
            arr = [float(v) for v in values]
            agent_data.metrics[f"{key}_count"] = len(arr)
            agent_data.metrics[f"{key}_min"] = min(arr)
            agent_data.metrics[f"{key}_max"] = max(arr)
            agent_data.metrics[f"{key}_mean"] = sum(arr) / len(arr)

    def _finalize_speculative_metrics(self, agent_data: AgentData) -> None:
        records = agent_data.speculative_records
        if not records:
            return
        totals = [r for r in records if r.get("main_tool_calls")]
        matches = [r for r in totals if r.get("tool_call_match")]
        reused = [r for r in records if r.get("prefetch_reused")]
        fallbacks = [r for r in totals if not r.get("tool_call_match")]
        agent_data.metrics["speculative_turns"] = len(records)
        agent_data.metrics["speculative_tool_turns"] = len(totals)
        agent_data.metrics["speculative_matches"] = len(matches)
        agent_data.metrics["speculative_reused"] = len(reused)
        agent_data.metrics["speculative_fallbacks"] = len(fallbacks)
        agent_data.metrics["speculative_match_rate"] = len(matches) / len(totals) if totals else 0.0
        deltas = [float(r["prefetch_delta_after_main_s"]) for r in records if "prefetch_delta_after_main_s" in r]
        if deltas:
            agent_data.metrics["speculative_prefetch_delta_after_main_mean"] = sum(deltas) / len(deltas)
            agent_data.metrics["speculative_prefetch_delta_after_main_min"] = min(deltas)
            agent_data.metrics["speculative_prefetch_delta_after_main_max"] = max(deltas)

    def _write_speculative_jsonl(self, agent_data: AgentData) -> None:
        if not self.speculative_jsonl or not agent_data.speculative_records:
            return
        os.makedirs(os.path.dirname(self.speculative_jsonl) or ".", exist_ok=True)
        with open(self.speculative_jsonl, "a", encoding="utf-8") as fout:
            fout.write(
                json.dumps(
                    {
                        "event": "hotpot_speculative_sample",
                        "request_id": agent_data.request_id,
                        "tool_call_count": agent_data.tool_call_count,
                        "records": agent_data.speculative_records,
                        "tool_latency_s": agent_data.metrics.get("tool_latency_s", []),
                        "tool_configured_sleep_s": agent_data.metrics.get("tool_configured_sleep_s", []),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
