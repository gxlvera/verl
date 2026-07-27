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

"""Transport-neutral Agent Service Proxy contracts and in-memory state."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from typing import Any, Literal

CANONICALIZATION_VERSION = "v0"


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain only JSON-compatible values") from exc


def _copy_mapping_sequence(
    value: Sequence[Mapping[str, Any]],
    field_name: str,
) -> tuple[dict[str, Any], ...]:
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{field_name} must be a sequence of mappings")
    return tuple(_copy_mapping(item, f"{field_name}[{index}]") for index, item in enumerate(value))


def _validate_token_ids(value: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{field_name} must be a sequence of token IDs")
    result: list[int] = []
    for index, token_id in enumerate(value):
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"{field_name}[{index}] must be a non-negative integer")
        result.append(token_id)
    return tuple(result)


def _to_wire(value: Any) -> Any:
    if isinstance(value, Enum):
        return _to_wire(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _to_wire(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _to_wire(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_wire(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_wire(value.tolist())
    return value


class FrontendProtocol(StrEnum):
    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_RESPONSES = "openai_responses"


class SessionLifecycleState(StrEnum):
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    FINALIZING = "FINALIZING"
    FINALIZED = "FINALIZED"
    ABORTED = "ABORTED"


class ChainLifecycleState(StrEnum):
    ACTIVE = "ACTIVE"
    RESERVED = "RESERVED"
    COMPLETED = "COMPLETED"


class DeliveryStatus(StrEnum):
    DELIVERED = "delivered"
    FAILED = "failed"
    UNKNOWN = "unknown"


class SplitReason(StrEnum):
    MESSAGE_PREFIX_DIVERGED = "message_prefix_diverged"
    TOOLS_CHANGED = "tools_changed"
    CONTEXT_COMPACTION = "context_compaction"
    REPEATED_PROMPT_RESAMPLE = "repeated_prompt_resample"
    AMBIGUOUS_PARENT = "ambiguous_parent"


TokenProvenanceKind = Literal["prompt", "generated", "replayed_context"]


@dataclass(frozen=True)
class AgentSessionHandle:
    session_id: str
    frontend_endpoint: str
    auth_token: str
    model_name: str | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "frontend_endpoint", "auth_token"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class GenerationSpec:
    sampling_params: Mapping[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 1024
    max_model_requests: int = 64
    max_total_tokens: int = 131072

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampling_params", _copy_mapping(self.sampling_params, "sampling_params"))
        for name in ("max_new_tokens", "max_model_requests", "max_total_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CanonicalMessage:
    """Immutable canonical message represented by stable JSON."""

    canonical_json: str

    def __post_init__(self) -> None:
        value = self.to_dict()
        role = value.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("canonical message role must be a non-empty string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalMessage:
        copied = _copy_mapping(value, "canonical message")
        return cls(json.dumps(copied, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))

    @property
    def role(self) -> str:
        return self.to_dict()["role"]

    def to_dict(self) -> dict[str, Any]:
        try:
            value = json.loads(self.canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("canonical_json must encode a JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("canonical_json must encode a JSON object")
        return value


@dataclass(frozen=True)
class CanonicalTool:
    """Immutable canonical tool schema represented by stable JSON."""

    canonical_json: str

    def __post_init__(self) -> None:
        self.to_dict()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CanonicalTool:
        copied = _copy_mapping(value, "canonical tool")
        return cls(json.dumps(copied, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        try:
            value = json.loads(self.canonical_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("canonical_json must encode a JSON object") from exc
        if not isinstance(value, dict):
            raise ValueError("canonical_json must encode a JSON object")
        return value


@dataclass(frozen=True)
class CanonicalPrompt:
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[CanonicalTool, ...]
    tools_fingerprint: str
    canonicalization_version: str = CANONICALIZATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.tools_fingerprint, str) or not self.tools_fingerprint:
            raise ValueError("tools_fingerprint must be a non-empty string")


@dataclass(frozen=True)
class MessagePrefix:
    message_count: int
    rolling_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.message_count, int) or isinstance(self.message_count, bool) or self.message_count < 0:
            raise ValueError("message_count must be a non-negative integer")
        if not isinstance(self.rolling_hash, str) or not self.rolling_hash:
            raise ValueError("rolling_hash must be a non-empty string")


@dataclass(frozen=True)
class InternalGenerationRequest:
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    sampling_params: Mapping[str, Any]
    stream: bool
    protocol: FrontendProtocol
    request_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", _copy_mapping_sequence(self.messages, "messages"))
        object.__setattr__(self, "tools", _copy_mapping_sequence(self.tools, "tools"))
        object.__setattr__(self, "sampling_params", _copy_mapping(self.sampling_params, "sampling_params"))
        object.__setattr__(
            self,
            "request_metadata",
            _copy_mapping(self.request_metadata, "request_metadata"),
        )
        if not isinstance(self.stream, bool):
            raise TypeError("stream must be a boolean")
        if not isinstance(self.protocol, FrontendProtocol):
            object.__setattr__(self, "protocol", FrontendProtocol(self.protocol))


@dataclass(frozen=True)
class TurnRecord:
    turn_id: str
    source_turn_id: str
    chain_id: str
    segment_id: str
    request_messages: tuple[CanonicalMessage, ...]
    request_tools: tuple[CanonicalTool, ...]
    tools_fingerprint: str
    backend_prompt_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...] | None
    generation_mask: tuple[int, ...]
    routed_experts: Any | None
    response_message: CanonicalMessage
    finish_reason: str
    upstream_replica: str
    delivery_status: DeliveryStatus
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("turn_id", "source_turn_id", "chain_id", "segment_id", "tools_fingerprint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "backend_prompt_ids",
            _validate_token_ids(self.backend_prompt_ids, "backend_prompt_ids"),
        )
        object.__setattr__(
            self,
            "output_token_ids",
            _validate_token_ids(self.output_token_ids, "output_token_ids"),
        )
        if self.output_logprobs is not None:
            logprobs = tuple(float(value) for value in self.output_logprobs)
            if len(logprobs) != len(self.output_token_ids):
                raise ValueError("output_logprobs length must match output_token_ids")
            if any(not math.isfinite(value) for value in logprobs):
                raise ValueError("output_logprobs values must be finite")
            object.__setattr__(self, "output_logprobs", logprobs)
        generation_mask = tuple(self.generation_mask)
        if len(generation_mask) != len(self.output_token_ids) or any(value not in (0, 1) for value in generation_mask):
            raise ValueError("generation_mask must be binary and align with output_token_ids")
        object.__setattr__(self, "generation_mask", generation_mask)
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "source_turn_id": self.source_turn_id,
            "chain_id": self.chain_id,
            "segment_id": self.segment_id,
            "request_messages": [message.to_dict() for message in self.request_messages],
            "request_tools": [tool.to_dict() for tool in self.request_tools],
            "tools_fingerprint": self.tools_fingerprint,
            "backend_prompt_ids": list(self.backend_prompt_ids),
            "output_token_ids": list(self.output_token_ids),
            "output_logprobs": list(self.output_logprobs) if self.output_logprobs is not None else None,
            "generation_mask": list(self.generation_mask),
            "routed_experts": _to_wire(self.routed_experts),
            "response_message": self.response_message.to_dict(),
            "finish_reason": self.finish_reason,
            "upstream_replica": self.upstream_replica,
            "delivery_status": self.delivery_status.value,
            "metadata": _to_wire(self.metadata),
        }


@dataclass(frozen=True)
class BoundaryAdjustment:
    reason: Literal["model_boundary_prefix_removal"]
    request_id: str
    turn_id: str
    removed_prefix_token_ids: tuple[int, ...]
    removed_generation_mask: tuple[int, ...]
    removed_logprobs: tuple[float, ...] | None
    removed_source_turn_ids: tuple[str | None, ...]
    removed_provenance_kinds: tuple[TokenProvenanceKind, ...]

    def __post_init__(self) -> None:
        if self.reason != "model_boundary_prefix_removal":
            raise ValueError("Unsupported boundary adjustment reason")
        for name in ("request_id", "turn_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "removed_prefix_token_ids",
            _validate_token_ids(self.removed_prefix_token_ids, "removed_prefix_token_ids"),
        )
        token_count = len(self.removed_prefix_token_ids)
        aligned = {
            "removed_generation_mask": len(self.removed_generation_mask),
            "removed_source_turn_ids": len(self.removed_source_turn_ids),
            "removed_provenance_kinds": len(self.removed_provenance_kinds),
        }
        if self.removed_logprobs is not None:
            aligned["removed_logprobs"] = len(self.removed_logprobs)
        mismatched = {name: length for name, length in aligned.items() if length != token_count}
        if mismatched:
            raise ValueError(f"Boundary adjustment arrays do not align: {mismatched}")
        if any(value not in (0, 1) for value in self.removed_generation_mask):
            raise ValueError("removed_generation_mask must be binary")
        object.__setattr__(self, "removed_generation_mask", tuple(self.removed_generation_mask))
        object.__setattr__(self, "removed_source_turn_ids", tuple(self.removed_source_turn_ids))
        object.__setattr__(self, "removed_provenance_kinds", tuple(self.removed_provenance_kinds))
        if self.removed_logprobs is not None:
            logprobs = tuple(float(value) for value in self.removed_logprobs)
            if any(not math.isfinite(value) for value in logprobs):
                raise ValueError("removed_logprobs values must be finite")
            object.__setattr__(self, "removed_logprobs", logprobs)

    def to_dict(self) -> dict[str, Any]:
        return _to_wire(self)


@dataclass
class TokenSegmentState:
    segment_id: str
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_generation_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = field(default_factory=list)
    source_turn_ids: list[str | None] = field(default_factory=list)
    provenance_kinds: list[TokenProvenanceKind] = field(default_factory=list)
    split_reason: SplitReason | None = None
    boundary_adjustments: list[BoundaryAdjustment] = field(default_factory=list)

    @property
    def exact_token_ids(self) -> tuple[int, ...]:
        return tuple(self.prompt_ids + self.response_ids)

    def validate(self) -> None:
        _validate_token_ids(self.prompt_ids, "prompt_ids")
        _validate_token_ids(self.response_ids, "response_ids")
        response_length = len(self.response_ids)
        aligned = {
            "response_generation_mask": len(self.response_generation_mask),
            "source_turn_ids": len(self.source_turn_ids),
            "provenance_kinds": len(self.provenance_kinds),
        }
        if self.response_logprobs is not None:
            aligned["response_logprobs"] = len(self.response_logprobs)
        mismatched = {name: length for name, length in aligned.items() if length != response_length}
        if mismatched:
            raise ValueError(f"Token segment arrays do not align with response_ids: {mismatched}")
        if any(value not in (0, 1) for value in self.response_generation_mask):
            raise ValueError("response_generation_mask must be binary")
        for index, adjustment in enumerate(self.boundary_adjustments):
            if not isinstance(adjustment, BoundaryAdjustment):
                raise TypeError(f"boundary_adjustments[{index}] must be a BoundaryAdjustment")


@dataclass
class ChainState:
    chain_id: str
    parent_chain_id: str | None
    branch_message_index: int
    split_reason: SplitReason | None
    message_history: list[CanonicalMessage]
    message_prefix_hashes: list[str]
    tools: tuple[CanonicalTool, ...]
    tools_fingerprint: str
    segments: list[TokenSegmentState]
    turns: list[TurnRecord] = field(default_factory=list)
    state: ChainLifecycleState = ChainLifecycleState.ACTIVE
    updated_sequence: int = 0
    materialization_order: int = 0
    reserved_by: str | None = None
    pending_request_messages: tuple[CanonicalMessage, ...] | None = None

    @property
    def current_segment(self) -> TokenSegmentState:
        if not self.segments:
            raise ValueError(f"Chain {self.chain_id} has no token segment")
        return self.segments[-1]


@dataclass(frozen=True)
class SessionEvent:
    sequence: int
    event_type: str
    request_id: str | None
    chain_id: str | None
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if not isinstance(self.event_type, str) or not self.event_type:
            raise ValueError("event_type must be a non-empty string")
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))


@dataclass(frozen=True)
class TokenProvenance:
    source_turn_id: str | None
    kind: TokenProvenanceKind

    def to_dict(self) -> dict[str, Any]:
        return {"source_turn_id": self.source_turn_id, "kind": self.kind}


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    session_id: str
    chain_id: str
    segment_id: str
    parent_chain_id: str | None
    split_reason: SplitReason | None
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    generation_mask: tuple[int, ...]
    loss_mask: tuple[int, ...] | None
    loss_weight: tuple[float, ...] | None
    response_logprobs: tuple[float, ...] | None
    token_provenance: tuple[TokenProvenance, ...]
    routed_experts: Any | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "session_id", "chain_id", "segment_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "prompt_ids", _validate_token_ids(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "response_ids", _validate_token_ids(self.response_ids, "response_ids"))
        response_length = len(self.response_ids)
        aligned: dict[str, int] = {
            "generation_mask": len(self.generation_mask),
            "token_provenance": len(self.token_provenance),
        }
        for name in ("loss_mask", "loss_weight", "response_logprobs"):
            value = getattr(self, name)
            if value is not None:
                aligned[name] = len(value)
        mismatched = {name: length for name, length in aligned.items() if length != response_length}
        if mismatched:
            raise ValueError(f"Trajectory arrays do not align with response_ids: {mismatched}")
        if any(value not in (0, 1) for value in self.generation_mask):
            raise ValueError("generation_mask must be binary")
        object.__setattr__(self, "generation_mask", tuple(self.generation_mask))
        if self.loss_mask is not None and any(value not in (0, 1) for value in self.loss_mask):
            raise ValueError("loss_mask must be binary")
        if self.loss_mask is not None:
            object.__setattr__(self, "loss_mask", tuple(self.loss_mask))
        if self.loss_weight is not None:
            loss_weight = tuple(float(value) for value in self.loss_weight)
            if any(not math.isfinite(value) or value < 0 for value in loss_weight):
                raise ValueError("loss_weight values must be finite and non-negative")
            object.__setattr__(self, "loss_weight", loss_weight)
        if self.response_logprobs is not None:
            response_logprobs = tuple(float(value) for value in self.response_logprobs)
            if any(not math.isfinite(value) for value in response_logprobs):
                raise ValueError("response_logprobs values must be finite")
            object.__setattr__(self, "response_logprobs", response_logprobs)
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        payload = _to_wire(self)
        payload["num_turns"] = int(self.metadata.get("turn_count", 0))
        payload["response_mask"] = list(self.loss_mask if self.loss_mask is not None else self.generation_mask)
        payload["metrics"] = dict(self.metadata.get("metrics", {}))
        payload["extra_fields"] = dict(self.metadata.get("extra_fields", {}))
        return payload


@dataclass(frozen=True)
class TrajectoryBundle:
    session_id: str
    trajectories: tuple[Trajectory, ...]
    lineage: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        trajectory_ids = [trajectory.trajectory_id for trajectory in self.trajectories]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("trajectory IDs must be unique inside a bundle")
        if any(trajectory.session_id != self.session_id for trajectory in self.trajectories):
            raise ValueError("all trajectories must belong to the bundle session")
        object.__setattr__(self, "lineage", _copy_mapping(self.lineage, "lineage"))
        object.__setattr__(self, "metadata", _copy_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
            "lineage": _to_wire(self.lineage),
            "metadata": _to_wire(self.metadata),
        }
