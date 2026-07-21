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

"""Transport-neutral Agent Service task models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, NewType

from .errors import AgentServiceProtocolError

TaskId = NewType("TaskId", str)


class TaskStatus(str, Enum):
    """Server-owned Task state machine from the Agent Service V0 contract."""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    LAUNCHING = "LAUNCHING"
    RUNNING = "RUNNING"
    EXECUTION_FINISHED = "EXECUTION_FINISHED"
    FINALIZING_TRAJECTORY = "FINALIZING_TRAJECTORY"
    COMPUTING_REWARD = "COMPUTING_REWARD"
    CLEANING_UP = "CLEANING_UP"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in TERMINAL_TASK_STATUSES


TERMINAL_TASK_STATUSES = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED})


def _to_wire(value: Any) -> Any:
    """Convert SDK models to JSON-compatible values without mutating user input."""

    if isinstance(value, Enum):
        return _to_wire(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _to_wire(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object keys must be strings, got {type(key).__name__}")
            result[key] = _to_wire(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_wire(item) for item in value]
    raise TypeError(f"Value of type {type(value).__name__} is not JSON serializable")


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AgentServiceProtocolError(f"{field_name} must be a JSON object")
    return value


def _parse_datetime(value: Any, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentServiceProtocolError(f"{field_name} must be an ISO-8601 string or null")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise AgentServiceProtocolError(f"{field_name} is not a valid ISO-8601 timestamp: {value!r}") from exc


@dataclass(frozen=True)
class TaskSpec:
    """Public, per-sample Task specification sent by a training driver.

    The top-level ownership boundary is intentionally strict. ``sample_fields``
    is the open-ended data plane: it preserves the names of the per-sample
    ``DataProto.non_tensor_batch`` fields that a native AgentLoop would receive
    as keyword arguments. The canonical prompt is normalized separately to
    ``problem.messages`` rather than duplicated as ``sample_fields.raw_prompt``.
    Binary multimodal values remain at their original position and use an
    inline JSON media representation on the wire.

    Nested specs remain JSON objects for now so their schemas can evolve
    independently while the server implementation is being built.
    """

    problem: Mapping[str, Any]
    agent: Mapping[str, Any]
    execution: Mapping[str, Any]
    reward: Mapping[str, Any]
    generation: Mapping[str, Any]
    environment: Mapping[str, Any] | None = None
    lifecycle: Mapping[str, Any] | None = None
    sample_fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "problem": _to_wire(self.problem),
            "agent": _to_wire(self.agent),
            "execution": _to_wire(self.execution),
            "reward": _to_wire(self.reward),
            "generation": _to_wire(self.generation),
            "sample_fields": _to_wire(self.sample_fields),
        }
        if self.environment is not None:
            payload["environment"] = _to_wire(self.environment)
        if self.lifecycle is not None:
            payload["lifecycle"] = _to_wire(self.lifecycle)
        return payload


@dataclass(frozen=True)
class TaskError:
    """Structured error recorded in a terminal Task snapshot."""

    code: str
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskError:
        data = _require_mapping(value, "error")
        code = data.get("code")
        message = data.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            raise AgentServiceProtocolError("error.code and error.message must be strings")
        retryable = data.get("retryable", False)
        if not isinstance(retryable, bool):
            raise AgentServiceProtocolError("error.retryable must be a boolean")
        details = data.get("details", {})
        _require_mapping(details, "error.details")
        return cls(code=code, message=message, retryable=retryable, details=dict(details))


@dataclass(frozen=True)
class TaskSnapshot:
    """Point-in-time state and optional terminal result for one Task."""

    task_id: TaskId
    status: TaskStatus
    session_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    trajectory: Any = None
    reward: float | Mapping[str, Any] | None = None
    error: TaskError | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def final_reward(self) -> float | None:
        if isinstance(self.reward, int | float) and not isinstance(self.reward, bool):
            return float(self.reward)
        if isinstance(self.reward, Mapping):
            value = self.reward.get("final_reward", self.reward.get("reward"))
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
        return None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSnapshot:
        data = dict(_require_mapping(value, "task snapshot"))
        task_id = data.pop("task_id", None)
        status = data.pop("status", None)
        if not isinstance(task_id, str) or not task_id:
            raise AgentServiceProtocolError("task snapshot.task_id must be a non-empty string")
        if not isinstance(status, str):
            raise AgentServiceProtocolError("task snapshot.status must be a string")
        try:
            parsed_status = TaskStatus(status)
        except ValueError as exc:
            raise AgentServiceProtocolError(f"Unknown Task status: {status!r}") from exc

        session_id = data.pop("session_id", None)
        if session_id is not None and not isinstance(session_id, str):
            raise AgentServiceProtocolError("task snapshot.session_id must be a string or null")

        error_value = data.pop("error", None)
        error = TaskError.from_dict(error_value) if error_value is not None else None
        reward = data.pop("reward", None)
        if reward is not None and not (
            (isinstance(reward, int | float) and not isinstance(reward, bool)) or isinstance(reward, Mapping)
        ):
            raise AgentServiceProtocolError("task snapshot.reward must be a number, JSON object, or null")

        return cls(
            task_id=TaskId(task_id),
            status=parsed_status,
            session_id=session_id,
            created_at=_parse_datetime(data.pop("created_at", None), "task snapshot.created_at"),
            updated_at=_parse_datetime(data.pop("updated_at", None), "task snapshot.updated_at"),
            completed_at=_parse_datetime(data.pop("completed_at", None), "task snapshot.completed_at"),
            trajectory=data.pop("trajectory", None),
            reward=dict(reward) if isinstance(reward, Mapping) else reward,
            error=error,
            extra=data,
        )
