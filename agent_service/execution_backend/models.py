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

"""ExecutionBackend data contracts shared by controllers and implementations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _json_value(value: Any, field_name: str) -> Any:
    """Copy and validate one JSON-compatible value."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            result[key] = _json_value(item, f"{field_name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item, field_name) for item in value]
    raise TypeError(f"{field_name} must be JSON-compatible, got {type(value).__name__}")


def _validate_command(command: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(command, str | bytes) or not command:
        raise ValueError(f"{field_name} must contain at least one argument")
    normalized = tuple(command)
    if any(not isinstance(item, str) for item in normalized):
        raise TypeError(f"{field_name} arguments must be strings")
    return normalized


def _validate_timeout(value: float | None, field_name: str) -> None:
    if value is not None and (not isinstance(value, int | float) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{field_name} must be greater than zero or null")


class LocalRuntimeKind(str, Enum):
    """Agent execution strategy selected once for a LocalExecutionBackend."""

    COROUTINE = "coroutine"
    PROCESS = "process"


class ExecutionState(str, Enum):
    """Lifecycle facts reported by an ExecutionBackend."""

    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ExecutionState.SUCCEEDED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
            ExecutionState.TIMED_OUT,
        }


@dataclass(frozen=True)
class LocalEnvironmentSpec:
    """Local working directory and environment retained until Backend cleanup."""

    working_dir: str | None = None
    variables: Mapping[str, str] = field(default_factory=dict)
    inherit_parent_variables: bool = True

    def __post_init__(self) -> None:
        if self.working_dir is not None and (not isinstance(self.working_dir, str) or not self.working_dir):
            raise ValueError("environment.working_dir must be a non-empty string or null")
        if not isinstance(self.inherit_parent_variables, bool):
            raise TypeError("environment.inherit_parent_variables must be a boolean")
        variables = dict(self.variables)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in variables.items()):
            raise TypeError("environment.variables must map strings to strings")
        object.__setattr__(self, "variables", variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "working_dir": self.working_dir,
            "variables": dict(self.variables),
            "inherit_parent_variables": self.inherit_parent_variables,
        }


@dataclass(frozen=True)
class CoroutineLaunchSpec:
    """Importable async white-box entrypoint and its JSON-compatible payload.

    The entrypoint may be an async function accepting ``TaskExecutionSpec`` or
    a zero-argument class whose ``run`` method accepts ``TaskExecutionSpec``.
    """

    entrypoint: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.entrypoint, str) or not self.entrypoint:
            raise ValueError("runtime.entrypoint must be a non-empty string")
        object.__setattr__(self, "payload", _json_value(self.payload, "runtime.payload"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": LocalRuntimeKind.COROUTINE.value,
            "entrypoint": self.entrypoint,
            "payload": _json_value(self.payload, "runtime.payload"),
        }


@dataclass(frozen=True)
class ProcessLaunchSpec:
    """Black-box command executed in one subprocess for one Task."""

    command: Sequence[str]
    stdin: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _validate_command(self.command, "runtime.command"))
        if self.stdin is not None and not isinstance(self.stdin, str):
            raise TypeError("runtime.stdin must be a string or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": LocalRuntimeKind.PROCESS.value,
            "command": list(self.command),
            "stdin": self.stdin,
        }


RuntimeLaunchSpec = CoroutineLaunchSpec | ProcessLaunchSpec


@dataclass(frozen=True)
class TaskExecutionSpec:
    """Controller-resolved, Agent-visible launch specification for one Task."""

    runtime: RuntimeLaunchSpec
    environment: LocalEnvironmentSpec = field(default_factory=LocalEnvironmentSpec)
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, CoroutineLaunchSpec | ProcessLaunchSpec):
            raise TypeError("runtime must be a CoroutineLaunchSpec or ProcessLaunchSpec")
        if not isinstance(self.environment, LocalEnvironmentSpec):
            raise TypeError("environment must be a LocalEnvironmentSpec")
        _validate_timeout(self.timeout_seconds, "timeout_seconds")

    @property
    def runtime_kind(self) -> LocalRuntimeKind:
        if isinstance(self.runtime, CoroutineLaunchSpec):
            return LocalRuntimeKind.COROUTINE
        return LocalRuntimeKind.PROCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime.to_dict(),
            "environment": self.environment.to_dict(),
            "timeout_seconds": self.timeout_seconds,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CommandExecutionSpec:
    """Verifier command run in the Task environment before cleanup."""

    command: Sequence[str]
    stdin: str | None = None
    timeout_seconds: float | None = None
    working_dir: str | None = None
    environment_variables: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _validate_command(self.command, "command"))
        if self.stdin is not None and not isinstance(self.stdin, str):
            raise TypeError("stdin must be a string or null")
        _validate_timeout(self.timeout_seconds, "timeout_seconds")
        if self.working_dir is not None and (not isinstance(self.working_dir, str) or not self.working_dir):
            raise ValueError("working_dir must be a non-empty string or null")
        variables = dict(self.environment_variables)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in variables.items()):
            raise TypeError("environment_variables must map strings to strings")
        object.__setattr__(self, "environment_variables", variables)


@dataclass(frozen=True)
class ExecutionError:
    """Structured execution failure captured inside an ExecutionResult."""

    code: str
    message: str
    traceback: str | None = None


@dataclass(frozen=True)
class ExecutionResult:
    """Terminal facts for one AgentRuntime execution."""

    status: ExecutionState
    output: Any = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    error: ExecutionError | None = None

    def __post_init__(self) -> None:
        if not self.status.is_terminal:
            raise ValueError("ExecutionResult status must be terminal")


@dataclass(frozen=True)
class CommandResult:
    """Result of execute_in_environment."""

    status: ExecutionState
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    error: ExecutionError | None = None

    def __post_init__(self) -> None:
        if not self.status.is_terminal:
            raise ValueError("CommandResult status must be terminal")


@dataclass(frozen=True)
class LaunchAck:
    """Acknowledgement for a new or idempotently repeated launch."""

    session_id: str
    created: bool


@dataclass(frozen=True)
class ExecutionSnapshot:
    """Non-blocking observation of Backend-owned execution state."""

    session_id: str
    state: ExecutionState
    result: ExecutionResult | None = None
