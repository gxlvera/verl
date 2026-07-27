# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from dataclasses import dataclass, field

import pytest

from agent_service.proxy import (
    Canonicalizer,
    CanonicalMessage,
    ContinuousTokenCodec,
    ContinuousTokenMergeError,
    DeliveryStatus,
    FrontendProtocol,
    GenerationSpec,
    InternalGenerationRequest,
    ProxySessionManager,
    SessionConflictError,
    SessionInactiveError,
    SplitReason,
    TokenGenerationOutput,
)


@dataclass
class _MergeResult:
    token_ids: list[int]
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class _Builder:
    def build_initial_tokens(self, messages, *, tools=None):
        return [10, len(messages)]

    def merge_non_assistant_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools=None):
        return _MergeResult(runtime_token_ids + [20 + len(updated_messages)])

    def merge_assistant_tokens(self, runtime_token_ids, assistant_token_ids):
        return _MergeResult(runtime_token_ids + assistant_token_ids)


def _request(messages):
    return InternalGenerationRequest(
        messages=tuple(messages),
        tools=(),
        sampling_params={"temperature": 0.5},
        stream=False,
        protocol=FrontendProtocol.OPENAI_CHAT_COMPLETIONS,
    )


async def _manager():
    return ProxySessionManager(
        replica_endpoints=["http://replica-a", "http://replica-b"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(_Builder()),
        replica_picker=lambda endpoints: endpoints[1],
        token_factory=lambda: "secret-token",
        model_name="policy",
    )


@pytest.mark.asyncio
async def test_session_create_is_idempotent_and_sticky():
    manager = await _manager()
    spec = GenerationSpec(max_new_tokens=8, max_model_requests=4, max_total_tokens=100)

    first = await manager.create_session(session_id="session-1", generation_spec=spec)
    second = await manager.create_session(session_id="session-1", generation_spec=spec)
    session = await manager.authorize("session-1", "secret-token")

    assert first == second
    assert session.upstream_replica == "http://replica-b"
    assert first.frontend_endpoint == "http://proxy/sessions/session-1"

    with pytest.raises(SessionConflictError):
        await manager.create_session(
            session_id="session-1",
            generation_spec=GenerationSpec(max_new_tokens=9),
        )


@pytest.mark.asyncio
async def test_generation_spec_normalizes_max_new_tokens_for_rollout_backend():
    manager = await _manager()
    await manager.create_session(
        session_id="session-1",
        generation_spec=GenerationSpec(
            sampling_params={"max_new_tokens": 6, "top_k": -1},
            max_new_tokens=8,
        ),
    )
    session = await manager.get_session("session-1")
    request = _request([{"role": "user", "content": "one"}])
    prepared = await session.prepare_turn(
        request_id="request-1",
        request=request,
        prompt=Canonicalizer().canonicalize(request),
    )

    assert prepared.sampling_params["max_tokens"] == 6
    assert "max_new_tokens" not in prepared.sampling_params
    assert prepared.sampling_params["top_k"] == -1
    await session.fail_turn(prepared, failure_reason="test", upstream_called=False)


@pytest.mark.asyncio
async def test_linear_turn_commits_exact_tokens_and_resumes_same_chain():
    manager = await _manager()
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    canonicalizer = Canonicalizer()
    first_request = _request([{"role": "user", "content": "one"}])
    first_prompt = canonicalizer.canonicalize(first_request)
    first = await session.prepare_turn(request_id="request-1", request=first_request, prompt=first_prompt)
    await session.commit_turn(
        first,
        output=TokenGenerationOutput(token_ids=(90,), logprobs=(-0.1,), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "two"}),
        delivery_status=DeliveryStatus.DELIVERED,
    )

    second_request = _request(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    )
    second = await session.prepare_turn(
        request_id="request-2",
        request=second_request,
        prompt=canonicalizer.canonicalize(second_request),
    )

    assert second.chain_id == first.chain_id
    assert second.context_ids[: len(first.context_ids) + 1] == (*first.context_ids, 90)

    await session.commit_turn(
        second,
        output=TokenGenerationOutput(token_ids=(91, 92), logprobs=(-0.2, -0.3), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "four"}),
    )
    chain = session.chains[first.chain_id]
    assert chain.current_segment.exact_token_ids == (*first.context_ids, 90, 23, 91, 92)
    assert chain.current_segment.response_generation_mask == [1, 0, 1, 1]
    assert chain.current_segment.source_turn_ids == [first.turn_id, None, second.turn_id, second.turn_id]


@pytest.mark.asyncio
async def test_declared_glm_style_boundary_removal_preserves_raw_generation_fact():
    class _RemovingBoundaryBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            return _MergeResult(
                runtime_token_ids[:-1] + [23],
                removed_prefix_token_count=1,
            )

    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(_RemovingBoundaryBuilder()),
        token_factory=lambda: "secret-token",
    )
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    canonicalizer = Canonicalizer()
    first_request = _request([{"role": "user", "content": "one"}])
    first = await session.prepare_turn(
        request_id="request-1",
        request=first_request,
        prompt=canonicalizer.canonicalize(first_request),
    )
    first_turn = await session.commit_turn(
        first,
        output=TokenGenerationOutput(token_ids=(777,), logprobs=(-0.1,), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "two"}),
    )
    second_request = _request(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ]
    )
    second = await session.prepare_turn(
        request_id="request-2",
        request=second_request,
        prompt=canonicalizer.canonicalize(second_request),
    )

    assert second.context_ids == (*first.context_ids, 23)
    assert second.removed_prefix_ids == (777,)

    await session.commit_turn(
        second,
        output=TokenGenerationOutput(token_ids=(91,), logprobs=(-0.2,), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "four"}),
    )
    chain = session.chains[first.chain_id]
    segment = chain.current_segment
    assert first_turn.output_token_ids == (777,)
    assert chain.turns[1].backend_prompt_ids == second.context_ids
    assert chain.turns[1].metadata["continuous_token_boundary_adjustment"]["removed_prefix_token_ids"] == [777]
    assert segment.response_ids == [23, 91]
    assert segment.response_generation_mask == [0, 1]
    assert [adjustment.to_dict() for adjustment in segment.boundary_adjustments] == [
        {
            "reason": "model_boundary_prefix_removal",
            "request_id": "request-2",
            "turn_id": second.turn_id,
            "removed_prefix_token_ids": [777],
            "removed_generation_mask": [1],
            "removed_logprobs": [-0.1],
            "removed_source_turn_ids": [first.turn_id],
            "removed_provenance_kinds": ["generated"],
        }
    ]


@pytest.mark.asyncio
async def test_boundary_removal_cannot_reach_into_the_initial_prompt():
    class _RemovingBoundaryBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            return _MergeResult(runtime_token_ids[:-1] + [23], removed_prefix_token_count=1)

    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(_RemovingBoundaryBuilder()),
        token_factory=lambda: "secret-token",
    )
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    canonicalizer = Canonicalizer()
    first_request = _request([{"role": "user", "content": "one"}])
    first = await session.prepare_turn(
        request_id="request-1",
        request=first_request,
        prompt=canonicalizer.canonicalize(first_request),
    )
    await session.commit_turn(
        first,
        output=TokenGenerationOutput(token_ids=(), logprobs=(), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": ""}),
    )
    second_request = _request(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "three"},
        ]
    )

    with pytest.raises(ContinuousTokenMergeError, match="immutable initial prompt"):
        await session.prepare_turn(
            request_id="request-2",
            request=second_request,
            prompt=canonicalizer.canonicalize(second_request),
        )

    assert session.chains[first.chain_id].current_segment.exact_token_ids == first.context_ids
    assert session.model_request_count == 1


@pytest.mark.asyncio
async def test_concurrent_identical_prompts_reserve_distinct_siblings():
    manager = await _manager()
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    request = _request([{"role": "user", "content": "same"}])
    prompt = Canonicalizer().canonicalize(request)

    first, second = await asyncio.gather(
        session.prepare_turn(request_id="request-1", request=request, prompt=prompt),
        session.prepare_turn(request_id="request-2", request=request, prompt=prompt),
    )

    assert first.chain_id != second.chain_id
    assert session.chains[second.chain_id].split_reason is SplitReason.REPEATED_PROMPT_RESAMPLE

    await asyncio.gather(
        session.fail_turn(first, failure_reason="test", upstream_called=False),
        session.fail_turn(second, failure_reason="test", upstream_called=False),
    )
    assert not session.in_flight_request_ids


@pytest.mark.asyncio
async def test_finalization_waits_for_inflight_and_rejects_new_requests():
    manager = await _manager()
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    request = _request([{"role": "user", "content": "one"}])
    prompt = Canonicalizer().canonicalize(request)
    prepared = await session.prepare_turn(request_id="request-1", request=request, prompt=prompt)

    finalizing = asyncio.create_task(session.begin_finalization(timeout_seconds=1))
    await asyncio.sleep(0)
    with pytest.raises(SessionInactiveError):
        await session.prepare_turn(request_id="request-2", request=request, prompt=prompt)
    await session.fail_turn(prepared, failure_reason="test", upstream_called=False)

    chains = await finalizing
    assert isinstance(chains, tuple)


@pytest.mark.asyncio
async def test_failed_provisional_chain_is_removed_when_unreferenced():
    manager = await _manager()
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    request = _request([{"role": "user", "content": "one"}])
    prepared = await session.prepare_turn(
        request_id="request-1",
        request=request,
        prompt=Canonicalizer().canonicalize(request),
    )

    await session.fail_turn(prepared, failure_reason="upstream", upstream_called=True)

    assert prepared.chain_id not in session.chains
    assert session.total_token_count == len(prepared.context_ids)


@pytest.mark.asyncio
async def test_merge_failure_happens_before_inference_and_leaves_chain_unchanged():
    class _ToggleBuilder(_Builder):
        fail = False

        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            if self.fail:
                raise ValueError("suffix merge failed")
            return super().merge_non_assistant_tokens(
                previous_messages,
                updated_messages,
                runtime_token_ids,
                tools=tools,
            )

    builder = _ToggleBuilder()
    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(builder),
        token_factory=lambda: "secret-token",
    )
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    canonicalizer = Canonicalizer()
    first_request = _request([{"role": "user", "content": "one"}])
    first = await session.prepare_turn(
        request_id="request-1",
        request=first_request,
        prompt=canonicalizer.canonicalize(first_request),
    )
    await session.commit_turn(
        first,
        output=TokenGenerationOutput(token_ids=(777,), logprobs=(-0.1,), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "decoded"}),
    )
    chain = session.chains[first.chain_id]
    exact_before = chain.current_segment.exact_token_ids
    history_before = tuple(chain.message_history)
    turn_count_before = len(chain.turns)
    request_count_before = session.model_request_count
    builder.fail = True
    next_request = _request(
        [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "decoded"},
            {"role": "user", "content": "next"},
        ]
    )

    with pytest.raises(ContinuousTokenMergeError):
        await session.prepare_turn(
            request_id="request-2",
            request=next_request,
            prompt=canonicalizer.canonicalize(next_request),
        )

    assert chain.current_segment.exact_token_ids == exact_before
    assert chain.current_segment.exact_token_ids[-1] == 777
    assert tuple(chain.message_history) == history_before
    assert len(chain.turns) == turn_count_before
    assert session.model_request_count == request_count_before
    assert len(session.chains) == 1
    assert session.events[-1].failure_reason == "continuous_token_merge_failed"
    assert chain.split_reason is None
