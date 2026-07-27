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

"""Materialize every generated chain without re-running a tokenizer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from enum import StrEnum
from typing import Any

from .models import TokenProvenance, Trajectory, TrajectoryBundle
from .session import ProxySession


class LossMaterializationPolicy(StrEnum):
    PER_LEAF = "per_leaf"
    BY_SOURCE_OCCURRENCE = "by_source_occurrence"
    WEIGHTED_PER_LEAF = "weighted_per_leaf"


def _routing_sequence_length(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except (IndexError, TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value)
    return None


class TrajectoryMaterializer:
    def __init__(
        self,
        *,
        loss_policy: LossMaterializationPolicy | str = LossMaterializationPolicy.PER_LEAF,
    ) -> None:
        self.loss_policy = LossMaterializationPolicy(loss_policy)

    def materialize(self, session: ProxySession) -> TrajectoryBundle:
        trajectories: list[Trajectory] = []
        chains = sorted(
            session.chains.values(),
            key=lambda chain: (chain.materialization_order, chain.chain_id),
        )
        for chain in chains:
            if not chain.turns:
                continue
            for segment_index, segment in enumerate(chain.segments):
                segment.validate()
                if not segment.response_ids:
                    continue
                segment_turns = [turn for turn in chain.turns if turn.segment_id == segment.segment_id]
                routed_experts = segment_turns[-1].routed_experts if segment_turns else None
                if any(turn.routed_experts is not None for turn in segment_turns) and routed_experts is None:
                    raise ValueError("Latest turn omitted routed_experts after an earlier turn provided them")
                if routed_experts is not None:
                    expected_routing_length = len(segment.prompt_ids) + len(segment.response_ids)
                    actual_routing_length = _routing_sequence_length(routed_experts)
                    if actual_routing_length != expected_routing_length:
                        raise ValueError(
                            "Latest routed_experts must align with the complete trajectory token sequence: "
                            f"expected {expected_routing_length}, got {actual_routing_length}"
                        )
                trajectories.append(
                    Trajectory(
                        trajectory_id=f"trajectory:{session.session_id}:{chain.chain_id}:{segment.segment_id}",
                        session_id=session.session_id,
                        chain_id=chain.chain_id,
                        segment_id=segment.segment_id,
                        parent_chain_id=chain.parent_chain_id,
                        split_reason=chain.split_reason,
                        prompt_ids=tuple(segment.prompt_ids),
                        response_ids=tuple(segment.response_ids),
                        generation_mask=tuple(segment.response_generation_mask),
                        loss_mask=tuple(segment.response_generation_mask),
                        loss_weight=None,
                        response_logprobs=(
                            tuple(segment.response_logprobs) if segment.response_logprobs is not None else None
                        ),
                        token_provenance=tuple(
                            TokenProvenance(source_turn_id, kind)
                            for source_turn_id, kind in zip(
                                segment.source_turn_ids,
                                segment.provenance_kinds,
                                strict=True,
                            )
                        ),
                        routed_experts=routed_experts,
                        metadata={
                            "canonicalization_version": session.canonicalizer.version,
                            "materialization_order": chain.materialization_order,
                            "segment_order": segment_index,
                            "turn_count": len(segment_turns),
                            "routed_expert_turn_ids": [
                                turn.turn_id for turn in segment_turns if turn.routed_experts is not None
                            ],
                            "turn_records": [turn.to_dict() for turn in segment_turns],
                            "boundary_adjustments": [
                                adjustment.to_dict() for adjustment in segment.boundary_adjustments
                            ],
                            "extra_fields": dict(segment_turns[-1].metadata) if segment_turns else {},
                            "generated_token_count": sum(segment.response_generation_mask),
                            "removed_generated_token_count": sum(
                                sum(adjustment.removed_generation_mask) for adjustment in segment.boundary_adjustments
                            ),
                            "response_token_count": len(segment.response_ids),
                            "delivery_statuses": [
                                turn.delivery_status.value
                                for turn in chain.turns
                                if turn.segment_id == segment.segment_id
                            ],
                            "loss_materialization_policy": self.loss_policy.value,
                        },
                    )
                )

        trajectories = self._apply_loss_policy(trajectories)
        lineage = {
            "chains": [
                {
                    "chain_id": chain.chain_id,
                    "parent_chain_id": chain.parent_chain_id,
                    "branch_message_index": chain.branch_message_index,
                    "split_reason": chain.split_reason.value if chain.split_reason else None,
                    "state": chain.state.value,
                    "materialization_order": chain.materialization_order,
                    "segment_ids": [segment.segment_id for segment in chain.segments],
                    "turn_ids": [turn.turn_id for turn in chain.turns],
                }
                for chain in chains
            ]
        }
        return TrajectoryBundle(
            session_id=session.session_id,
            trajectories=tuple(trajectories),
            lineage=lineage,
            metadata={
                "canonicalization_version": session.canonicalizer.version,
                "trajectory_count": len(trajectories),
                "chain_count": len(chains),
                "model_request_count": session.model_request_count,
                "total_token_count": session.total_token_count,
                "upstream_replica": session.upstream_replica,
                "loss_materialization_policy": self.loss_policy.value,
                "event_count": len(session.events),
                "failed_request_count": sum(event.failure_reason is not None for event in session.events),
            },
        )

    def _apply_loss_policy(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        if self.loss_policy is LossMaterializationPolicy.PER_LEAF:
            return trajectories

        source_occurrences: Counter[str] = Counter()
        for trajectory in trajectories:
            source_occurrences.update(
                {
                    provenance.source_turn_id
                    for generated, provenance in zip(
                        trajectory.generation_mask,
                        trajectory.token_provenance,
                        strict=True,
                    )
                    if generated and provenance.source_turn_id is not None
                }
            )

        if self.loss_policy is LossMaterializationPolicy.WEIGHTED_PER_LEAF:
            return [
                replace(
                    trajectory,
                    loss_weight=tuple(
                        (
                            1.0 / source_occurrences[provenance.source_turn_id]
                            if generated and provenance.source_turn_id is not None
                            else 0.0
                        )
                        for generated, provenance in zip(
                            trajectory.generation_mask,
                            trajectory.token_provenance,
                            strict=True,
                        )
                    ),
                )
                for trajectory in trajectories
            ]

        seen_sources: set[str] = set()
        result: list[Trajectory] = []
        for trajectory in trajectories:
            trajectory_sources = {
                provenance.source_turn_id
                for generated, provenance in zip(
                    trajectory.generation_mask,
                    trajectory.token_provenance,
                    strict=True,
                )
                if generated and provenance.source_turn_id is not None
            }
            trainable_sources = trajectory_sources - seen_sources
            result.append(
                replace(
                    trajectory,
                    loss_mask=tuple(
                        int(
                            bool(generated)
                            and provenance.source_turn_id is not None
                            and provenance.source_turn_id in trainable_sources
                        )
                        for generated, provenance in zip(
                            trajectory.generation_mask,
                            trajectory.token_provenance,
                            strict=True,
                        )
                    ),
                )
            )
            seen_sources.update(trajectory_sources)
        return result
