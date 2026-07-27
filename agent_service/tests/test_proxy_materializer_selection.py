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

from dataclasses import dataclass, field

import pytest

from agent_service.proxy import (
    Canonicalizer,
    CanonicalMessage,
    ContinuousTokenCodec,
    FrontendProtocol,
    GenerationSpec,
    InMemoryProxy,
    InternalGenerationRequest,
    ProxySessionManager,
    TokenGenerationOutput,
    TrajectoryMaterializer,
)
from agent_service.trajectory_selection import (
    AgentServiceTrajectoryFinalizer,
    TrajectorySelectionContext,
    TrajectorySelectionError,
    TrajectorySelectionSpec,
    TrajectorySelectorRegistry,
)


@dataclass
class _MergeResult:
    token_ids: list[int]
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class _Builder:
    def build_initial_tokens(self, messages, *, tools=None):
        return [10]

    def merge_non_assistant_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools=None):
        return _MergeResult(runtime_token_ids + [20])

    def merge_assistant_tokens(self, runtime_token_ids, assistant_token_ids):
        return _MergeResult(runtime_token_ids + assistant_token_ids)


def _request(content):
    return InternalGenerationRequest(
        messages=({"role": "user", "content": content},),
        tools=(),
        sampling_params={},
        stream=False,
        protocol=FrontendProtocol.OPENAI_CHAT_COMPLETIONS,
    )


async def _finalized_bundle():
    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(_Builder()),
        token_factory=lambda: "secret",
    )
    proxy = InMemoryProxy(session_manager=manager, materializer=TrajectoryMaterializer())
    await proxy.start()
    await proxy.create_session(session_id="session-1", generation_spec=GenerationSpec())
    session = await manager.get_session("session-1")
    canonicalizer = Canonicalizer()
    for index, token_ids in enumerate(((90,), (91, 92, 93))):
        request = _request("same")
        prepared = await session.prepare_turn(
            request_id=f"request-{index}",
            request=request,
            prompt=canonicalizer.canonicalize(request),
        )
        await session.commit_turn(
            prepared,
            output=TokenGenerationOutput(
                token_ids=token_ids,
                logprobs=tuple(-0.1 for _ in token_ids),
                finish_reason="stop",
                routed_experts=[[[token_index]] for token_index in range(1 + len(token_ids))],
            ),
            response_message=CanonicalMessage.from_dict({"role": "assistant", "content": f"a-{index}"}),
        )
    first = await proxy.finalize_session("session-1")
    second = await proxy.finalize_session("session-1")
    return first, second


@pytest.mark.asyncio
async def test_finalize_materializes_every_sibling_and_is_cached():
    first, second = await _finalized_bundle()

    assert first is second
    assert len(first.trajectories) == 2
    assert [sum(item.generation_mask) for item in first.trajectories] == [1, 3]
    assert [item.metadata["materialization_order"] for item in first.trajectories] == [0, 1]
    assert [len(item.routed_experts) for item in first.trajectories] == [2, 4]
    assert first.trajectories[0].metadata["turn_records"][0]["backend_prompt_ids"] == [10]
    assert first.trajectories[0].metadata["turn_records"][0]["output_token_ids"] == [90]
    assert len(first.lineage["chains"]) == 2


@pytest.mark.asyncio
async def test_materialization_retains_glm_style_removed_boundary_as_raw_fact():
    class _RemovingBoundaryBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            return _MergeResult(runtime_token_ids[:-1] + [20], removed_prefix_token_count=1)

    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(_RemovingBoundaryBuilder()),
        token_factory=lambda: "secret",
    )
    proxy = InMemoryProxy(session_manager=manager, materializer=TrajectoryMaterializer())
    await proxy.start()
    await proxy.create_session(session_id="session-glm", generation_spec=GenerationSpec())
    session = await manager.get_session("session-glm")
    canonicalizer = Canonicalizer()
    first_request = _request("one")
    first = await session.prepare_turn(
        request_id="request-1",
        request=first_request,
        prompt=canonicalizer.canonicalize(first_request),
    )
    await session.commit_turn(
        first,
        output=TokenGenerationOutput(token_ids=(90,), logprobs=(-0.1,), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "two"}),
    )
    second_request = InternalGenerationRequest(
        messages=(
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
        ),
        tools=(),
        sampling_params={},
        stream=False,
        protocol=FrontendProtocol.OPENAI_CHAT_COMPLETIONS,
    )
    second = await session.prepare_turn(
        request_id="request-2",
        request=second_request,
        prompt=canonicalizer.canonicalize(second_request),
    )
    await session.commit_turn(
        second,
        output=TokenGenerationOutput(token_ids=(91,), logprobs=(-0.2,), finish_reason="stop"),
        response_message=CanonicalMessage.from_dict({"role": "assistant", "content": "four"}),
    )

    bundle = await proxy.finalize_session("session-glm")
    trajectory = bundle.trajectories[0]
    assert trajectory.response_ids == (20, 91)
    assert trajectory.generation_mask == (0, 1)
    assert trajectory.metadata["turn_records"][0]["output_token_ids"] == [90]
    assert trajectory.metadata["boundary_adjustments"][0]["removed_prefix_token_ids"] == [90]
    assert trajectory.metadata["removed_generated_token_count"] == 1


@pytest.mark.asyncio
async def test_builtin_all_and_longest_create_views_without_mutating_raw_bundle():
    bundle, _ = await _finalized_bundle()
    registry = TrajectorySelectorRegistry()

    all_result = registry.select(bundle, spec=TrajectorySelectionSpec(strategy="all"))
    longest_result = registry.select(bundle)

    assert all_result.selected_trajectory_ids == tuple(trajectory.trajectory_id for trajectory in bundle.trajectories)
    assert longest_result.selected_trajectory_ids == (bundle.trajectories[1].trajectory_id,)
    assert len(bundle.trajectories) == 2
    assert "selection" not in bundle.metadata


@pytest.mark.asyncio
async def test_server_registered_custom_selector_uses_same_contract():
    bundle, _ = await _finalized_bundle()

    class _FirstSelector:
        name = "first"
        version = "test-v1"

        def select(self, bundle, context, config):
            assert context.task_metadata == {"task": "one"}
            assert config == {"enabled": True}
            return [bundle.trajectories[0].trajectory_id]

    registry = TrajectorySelectorRegistry()
    registry.register(_FirstSelector())
    result = registry.select(
        bundle,
        spec=TrajectorySelectionSpec(strategy="first", config={"enabled": True}),
        context=TrajectorySelectionContext(task_metadata={"task": "one"}),
    )

    assert result.selector_name == "first"
    assert result.selector_version == "test-v1"
    assert result.selected_trajectory_ids == (bundle.trajectories[0].trajectory_id,)


@pytest.mark.asyncio
async def test_invalid_selector_output_fails_without_changing_complete_bundle():
    bundle, _ = await _finalized_bundle()

    class _InvalidSelector:
        name = "invalid"
        version = "test-v1"

        def select(self, bundle, context, config):
            return ["not-in-bundle"]

    registry = TrajectorySelectorRegistry()
    registry.register(_InvalidSelector())
    original_ids = tuple(trajectory.trajectory_id for trajectory in bundle.trajectories)

    with pytest.raises(TrajectorySelectionError, match="unknown IDs"):
        registry.select(bundle, spec=TrajectorySelectionSpec(strategy="invalid"))

    assert tuple(trajectory.trajectory_id for trajectory in bundle.trajectories) == original_ids


def test_driver_strategy_cannot_resolve_an_import_path():
    registry = TrajectorySelectorRegistry()

    with pytest.raises(TrajectorySelectionError, match="Unknown server-registered"):
        registry.resolve("some.module:Selector")


@pytest.mark.asyncio
async def test_agent_service_finalizer_retains_complete_bundle_before_selection():
    bundle, _ = await _finalized_bundle()

    class _Proxy:
        async def finalize_session(self, session_id):
            assert session_id == bundle.session_id
            return bundle

    finalizer = AgentServiceTrajectoryFinalizer()
    result = await finalizer.finalize(proxy=_Proxy(), session_id=bundle.session_id)

    assert len(result.selected_bundle.trajectories) == 1
    assert finalizer.get_complete_bundle(bundle.session_id) is bundle
    assert len(finalizer.get_complete_bundle(bundle.session_id).trajectories) == 2
