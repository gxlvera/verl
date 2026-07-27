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

from agent_service.proxy import (
    CanonicalMessage,
    CanonicalPrompt,
    ChainRouter,
    ChainState,
    DeliveryStatus,
    RoutingAction,
    SplitReason,
    TokenSegmentState,
    TurnRecord,
)


def _message(role, content):
    return CanonicalMessage.from_dict({"role": role, "content": content})


def _prompt(messages, tools_fingerprint="tools-a"):
    return CanonicalPrompt(messages=tuple(messages), tools=(), tools_fingerprint=tools_fingerprint)


def _chain(
    chain_id,
    messages,
    *,
    tools_fingerprint="tools-a",
    updated_sequence=1,
    request_boundaries=(),
    reserved_by=None,
):
    chain = ChainState(
        chain_id=chain_id,
        parent_chain_id=None,
        branch_message_index=0,
        split_reason=None,
        message_history=list(messages),
        message_prefix_hashes=[],
        tools=(),
        tools_fingerprint=tools_fingerprint,
        segments=[TokenSegmentState(segment_id=f"segment-{chain_id}", prompt_ids=[1])],
        updated_sequence=updated_sequence,
        reserved_by=reserved_by,
    )
    for index, boundary in enumerate(request_boundaries):
        chain.turns.append(
            TurnRecord(
                turn_id=f"turn-{chain_id}-{index}",
                source_turn_id=f"turn-{chain_id}-{index}",
                chain_id=chain_id,
                segment_id=f"segment-{chain_id}",
                request_messages=tuple(boundary),
                request_tools=(),
                tools_fingerprint=tools_fingerprint,
                backend_prompt_ids=(1,),
                output_token_ids=(2,),
                output_logprobs=(-0.1,),
                generation_mask=(1,),
                routed_experts=None,
                response_message=_message("assistant", "answer"),
                finish_reason="stop",
                upstream_replica="replica",
                delivery_status=DeliveryStatus.DELIVERED,
            )
        )
    return chain


def test_router_reuses_deepest_exact_canonical_prefix():
    user = _message("user", "one")
    assistant = _message("assistant", "two")
    next_user = _message("user", "three")
    shallow = _chain("shallow", [user], updated_sequence=99)
    deep = _chain("deep", [user, assistant], updated_sequence=1)

    decision = ChainRouter().select(_prompt([user, assistant, next_user]), [shallow, deep])

    assert decision.action is RoutingAction.REUSE
    assert decision.chain_id == "deep"
    assert decision.branch_message_index == 2


def test_router_creates_sibling_when_tools_change():
    user = _message("user", "one")
    chain = _chain("chain-a", [user], tools_fingerprint="tools-a", request_boundaries=[[user]])

    decision = ChainRouter().select(_prompt([user], tools_fingerprint="tools-b"), [chain])

    assert decision.action is RoutingAction.CREATE
    assert decision.parent_chain_id == "chain-a"
    assert decision.split_reason is SplitReason.TOOLS_CHANGED


def test_router_records_tools_changed_when_the_request_also_appends_a_message():
    user = _message("user", "one")
    assistant = _message("assistant", "two")
    next_user = _message("user", "three")
    chain = _chain("chain-a", [user, assistant], tools_fingerprint="tools-a")

    decision = ChainRouter().select(
        _prompt([user, assistant, next_user], tools_fingerprint="tools-b"),
        [chain],
    )

    assert decision.action is RoutingAction.CREATE
    assert decision.parent_chain_id == "chain-a"
    assert decision.split_reason is SplitReason.TOOLS_CHANGED
    assert decision.branch_message_index == 2


def test_router_treats_completed_prompt_boundary_as_resample():
    user = _message("user", "one")
    assistant = _message("assistant", "two")
    chain = _chain("chain-a", [user, assistant], request_boundaries=[[user]])

    decision = ChainRouter().select(_prompt([user]), [chain])

    assert decision.action is RoutingAction.CREATE
    assert decision.split_reason is SplitReason.REPEATED_PROMPT_RESAMPLE
    assert decision.branch_message_index == 1


def test_router_never_reuses_reserved_chain():
    user = _message("user", "one")
    assistant = _message("assistant", "two")
    next_user = _message("user", "three")
    chain = _chain("chain-a", [user, assistant], reserved_by="request-in-flight")

    decision = ChainRouter().select(_prompt([user, assistant, next_user]), [chain])

    assert decision.action is RoutingAction.CREATE
    assert decision.chain_id is None


def test_router_marks_explicit_compaction_without_rendered_prefix_logic():
    system = _message("system", "rules")
    old_user = _message("user", "long history")
    assistant = _message("assistant", "answer")
    summary = _message("user", "summary")
    chain = _chain("chain-a", [system, old_user, assistant])

    decision = ChainRouter().select(_prompt([system, summary]), [chain], context_compaction=True)

    assert decision.split_reason is SplitReason.CONTEXT_COMPACTION
    assert decision.branch_message_index == 1
