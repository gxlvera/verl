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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..client_sdk.models import _to_wire

SUPPORTED_FRONTEND_PROTOCOLS = frozenset(
    {
        "openai_chat_completions",
        "openai_responses",
        "anthropic_messages",
    }
)
SUPPORTED_UPSTREAM_PROTOCOLS = SUPPORTED_FRONTEND_PROTOCOLS | {"generate"}


@dataclass(frozen=True)
class InferenceSpec:
    """Fixed rollout inference registry installed in the Proxy at startup."""

    replica_endpoints: Sequence[str]
    upstream_protocol: str

    def __post_init__(self) -> None:
        endpoints = tuple(self.replica_endpoints)
        if not endpoints or any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoints):
            raise ValueError("inference.replica_endpoints must contain at least one non-empty URL")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("inference.replica_endpoints must not contain duplicates")
        if self.upstream_protocol not in SUPPORTED_UPSTREAM_PROTOCOLS:
            supported = ", ".join(sorted(SUPPORTED_UPSTREAM_PROTOCOLS))
            raise ValueError(f"inference.upstream_protocol must be one of: {supported}")
        object.__setattr__(self, "replica_endpoints", endpoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica_endpoints": list(self.replica_endpoints),
            "upstream_protocol": self.upstream_protocol,
        }


@dataclass(frozen=True)
class AgentServiceStartupConfig:
    """Experiment-scoped Agent Service bootstrap configuration.

    This object is passed to the Service Actor at creation time. It is not sent
    through the Task API and deliberately contains no per-Task fields.
    """

    execution_backend: Mapping[str, Any]
    inference: InferenceSpec
    proxy: Mapping[str, Any] | None = None
    admission: Mapping[str, Any] | None = None
    default_lifecycle: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution_backend, Mapping) or not self.execution_backend:
            raise ValueError("execution_backend must be a non-empty mapping")
        backend_kind = self.execution_backend.get("kind")
        if backend_kind != "local":
            raise ValueError("Agent Service V0 supports only execution_backend.kind='local'")
        for name in ("proxy", "admission", "default_lifecycle"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or null")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "execution_backend": _to_wire(self.execution_backend),
            "inference": self.inference.to_dict(),
        }
        for name in ("proxy", "admission", "default_lifecycle"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = _to_wire(value)
        return payload


def validate_agent_service_config(config: Mapping[str, Any]) -> None:
    """Validate verl-side integration settings when Agent Service is enabled."""

    if not config.get("enabled", False):
        return

    ray_actor = config.get("ray_actor")
    if not isinstance(ray_actor, Mapping) or not ray_actor.get("actor_class"):
        raise ValueError("agent_service.ray_actor.actor_class is required when Agent Service is enabled")

    execution_backend = config.get("execution_backend")
    if not isinstance(execution_backend, Mapping) or execution_backend.get("kind") != "local":
        raise ValueError("Agent Service V0 requires agent_service.execution_backend.kind=local")

    if not isinstance(config.get("upstream_protocol"), str) or not config["upstream_protocol"]:
        raise ValueError("agent_service.upstream_protocol must be a non-empty string")

    if config["upstream_protocol"] not in SUPPORTED_UPSTREAM_PROTOCOLS:
        supported = ", ".join(sorted(SUPPORTED_UPSTREAM_PROTOCOLS))
        raise ValueError(f"agent_service.upstream_protocol must be one of: {supported}")

    max_in_flight = config.get("max_in_flight")
    if max_in_flight is not None and (not isinstance(max_in_flight, int) or max_in_flight <= 0):
        raise ValueError("agent_service.max_in_flight must be greater than zero or null")

    for name in (
        "ready_timeout_seconds",
        "wait_timeout_seconds",
        "shutdown_timeout_seconds",
        "rpc_timeout_seconds",
    ):
        value = config.get(name)
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"agent_service.{name} must be greater than zero")

    task = config.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("agent_service.task must be a mapping")
    agent = task.get("agent")
    if not isinstance(agent, Mapping) or not agent.get("artifact"):
        raise ValueError("agent_service.task.agent.artifact is required")
    if agent.get("frontend_protocol") not in SUPPORTED_FRONTEND_PROTOCOLS:
        supported = ", ".join(sorted(SUPPORTED_FRONTEND_PROTOCOLS))
        raise ValueError(f"agent_service.task.agent.frontend_protocol must be one of: {supported}")

    task_execution = task.get("execution")
    command = task_execution.get("command") if isinstance(task_execution, Mapping) else None
    if not isinstance(command, Sequence) or isinstance(command, str | bytes) or not command:
        raise ValueError("agent_service.task.execution.command is required")

    reward = task.get("reward")
    reward_function = reward.get("reward_function") if isinstance(reward, Mapping) else None
    if not isinstance(reward_function, Mapping) or not reward_function:
        raise ValueError("agent_service.task.reward.reward_function is required")
