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

"""Canonical message-prefix and tool-context lineage routing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from .canonical import common_prefix_length, exact_prefix_length
from .models import CanonicalPrompt, ChainLifecycleState, ChainState, SplitReason


class RoutingAction(StrEnum):
    REUSE = "reuse"
    CREATE = "create"


@dataclass(frozen=True)
class RoutingDecision:
    action: RoutingAction
    chain_id: str | None
    parent_chain_id: str | None
    branch_message_index: int
    split_reason: SplitReason | None
    diagnostics: dict[str, object]


def _reusable_prefix_length(chain: ChainState, prompt: CanonicalPrompt) -> int | None:
    if chain.tools_fingerprint != prompt.tools_fingerprint:
        return None
    if chain.state is not ChainLifecycleState.ACTIVE or chain.reserved_by is not None:
        return None
    prefix_length = exact_prefix_length(chain.message_history, prompt.messages)
    if prefix_length is None or prefix_length == len(prompt.messages):
        return None
    return prefix_length


def _is_repeated_prompt(chain: ChainState, prompt: CanonicalPrompt) -> bool:
    if chain.tools_fingerprint != prompt.tools_fingerprint:
        return False
    if chain.pending_request_messages == prompt.messages:
        return True
    return any(turn.request_messages == prompt.messages for turn in chain.turns)


def _same_messages_with_different_tools(chain: ChainState, prompt: CanonicalPrompt) -> bool:
    if chain.tools_fingerprint == prompt.tools_fingerprint:
        return False
    if exact_prefix_length(chain.message_history, prompt.messages) is not None:
        return True
    if chain.pending_request_messages == prompt.messages:
        return True
    return any(exact_prefix_length(turn.request_messages, prompt.messages) is not None for turn in chain.turns)


class ChainRouter:
    """Select a chain without rendering or tokenizing the request."""

    def select(
        self,
        prompt: CanonicalPrompt,
        chains: Iterable[ChainState],
        *,
        context_compaction: bool = False,
    ) -> RoutingDecision:
        chain_list = list(chains)
        reusable: list[tuple[int, int, str, ChainState]] = []
        for chain in chain_list:
            prefix_length = _reusable_prefix_length(chain, prompt)
            if prefix_length is not None:
                reusable.append((prefix_length, chain.updated_sequence, chain.chain_id, chain))
        if reusable:
            prefix_length, _, _, selected = max(reusable, key=lambda item: (item[0], item[1], item[2]))
            return RoutingDecision(
                action=RoutingAction.REUSE,
                chain_id=selected.chain_id,
                parent_chain_id=selected.parent_chain_id,
                branch_message_index=prefix_length,
                split_reason=None,
                diagnostics={
                    "matched_message_count": prefix_length,
                    "candidate_count": len(reusable),
                    "tools_fingerprint": prompt.tools_fingerprint,
                },
            )

        if not chain_list:
            return RoutingDecision(
                action=RoutingAction.CREATE,
                chain_id=None,
                parent_chain_id=None,
                branch_message_index=0,
                split_reason=None,
                diagnostics={"matched_message_count": 0, "candidate_count": 0},
            )

        lineage_candidates: list[tuple[int, int, str, ChainState]] = []
        for chain in chain_list:
            lineage_candidates.append(
                (
                    common_prefix_length(chain.message_history, prompt.messages),
                    chain.updated_sequence,
                    chain.chain_id,
                    chain,
                )
            )
        best_prefix = max(item[0] for item in lineage_candidates)
        deepest = [item for item in lineage_candidates if item[0] == best_prefix]
        _, _, _, parent = max(deepest, key=lambda item: (item[1], item[2]))

        repeated = [chain for chain in chain_list if _is_repeated_prompt(chain, prompt)]
        tools_changed = [chain for chain in chain_list if _same_messages_with_different_tools(chain, prompt)]
        if repeated:
            split_reason = SplitReason.REPEATED_PROMPT_RESAMPLE
            parent = max(repeated, key=lambda chain: (chain.updated_sequence, chain.chain_id))
            branch_message_index = len(prompt.messages)
        elif tools_changed:
            split_reason = SplitReason.TOOLS_CHANGED
            parent = max(tools_changed, key=lambda chain: (chain.updated_sequence, chain.chain_id))
            branch_message_index = common_prefix_length(parent.message_history, prompt.messages)
        elif context_compaction:
            split_reason = SplitReason.CONTEXT_COMPACTION
            branch_message_index = best_prefix
        elif len(deepest) > 1 and best_prefix > 0:
            split_reason = SplitReason.AMBIGUOUS_PARENT
            branch_message_index = best_prefix
        else:
            split_reason = SplitReason.MESSAGE_PREFIX_DIVERGED
            branch_message_index = best_prefix

        return RoutingDecision(
            action=RoutingAction.CREATE,
            chain_id=None,
            parent_chain_id=parent.chain_id if branch_message_index > 0 else None,
            branch_message_index=branch_message_index,
            split_reason=split_reason,
            diagnostics={
                "matched_message_count": branch_message_index,
                "candidate_count": len(deepest),
                "parent_chain_id": parent.chain_id if branch_message_index > 0 else None,
                "tools_fingerprint": prompt.tools_fingerprint,
            },
        )
