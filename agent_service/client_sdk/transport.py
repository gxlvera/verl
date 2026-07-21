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

"""Agent Service transport interfaces and implementations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import ray

from .errors import (
    AgentServiceClosedError,
    AgentServiceProtocolError,
)
from .models import TaskId, TaskSnapshot, TaskSpec


class TransportClient(Protocol):
    """Driver-to-Agent-Service transport protocol consumed by ``AgentExecutor``.

    V0 uses ``RayTransportClient`` and Ray actor RPC. The protocol deliberately
    leaves the deployment transport open: if Agent Service is later exposed as
    a FastAPI service, implement an ``HTTPTransportClient`` with the same
    submit/get_status/wait_any/cancel methods. ``AgentExecutor``, the rollout
    adapter, and Driver call sites remain unchanged.
    """

    def submit(self, task_spec: TaskSpec, idempotency_key: str | None = None) -> TaskId: ...

    def get_status(self, task_id: TaskId) -> TaskSnapshot: ...

    def wait_any(
        self,
        task_ids: Sequence[TaskId],
        timeout_seconds: float,
        max_results: int,
    ) -> list[TaskSnapshot]: ...

    def cancel(self, task_id: TaskId) -> TaskSnapshot: ...

    def close(self) -> None: ...


class RayTransportClient:
    """V0 Driver transport that invokes a named Agent Service Ray actor."""

    def __init__(
        self,
        endpoint: Mapping[str, Any],
        *,
        rpc_timeout_seconds: float = 30.0,
        long_poll_margin_seconds: float = 5.0,
    ) -> None:
        endpoint = dict(endpoint)
        unknown_fields = set(endpoint) - {"transport", "actor_name", "namespace"}
        if unknown_fields:
            raise ValueError(f"Unknown Agent Service endpoint fields: {sorted(unknown_fields)}")
        if endpoint.get("transport") != "ray":
            raise ValueError("Ray transport endpoint.transport must be 'ray'")
        actor_name = endpoint.get("actor_name")
        if not isinstance(actor_name, str) or not actor_name:
            raise ValueError("endpoint.actor_name must be a non-empty string")
        namespace = endpoint.get("namespace")
        if namespace is not None and (not isinstance(namespace, str) or not namespace):
            raise ValueError("endpoint.namespace must be a non-empty string or null")
        if rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be greater than zero")
        if long_poll_margin_seconds < 0:
            raise ValueError("long_poll_margin_seconds must not be negative")

        self._actor = ray.get_actor(actor_name, namespace=namespace)
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self._long_poll_margin_seconds = long_poll_margin_seconds
        self._closed = False

    def submit(self, task_spec: TaskSpec, idempotency_key: str | None = None) -> TaskId:
        self._ensure_open()
        response = ray.get(
            self._actor.submit.remote(task_spec.to_dict(), idempotency_key),
            timeout=self._rpc_timeout_seconds,
        )
        data = self._expect_object(response, "SubmitTask response")
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise AgentServiceProtocolError("SubmitTask response.task_id must be a non-empty string")
        return TaskId(task_id)

    def get_status(self, task_id: TaskId) -> TaskSnapshot:
        self._ensure_open()
        response = ray.get(
            self._actor.get_status.remote(str(task_id)),
            timeout=self._rpc_timeout_seconds,
        )
        return TaskSnapshot.from_dict(self._expect_object(response, "GetStatus response"))

    def wait_any(
        self,
        task_ids: Sequence[TaskId],
        timeout_seconds: float,
        max_results: int,
    ) -> list[TaskSnapshot]:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must not be negative")
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        self._ensure_open()
        response = ray.get(
            self._actor.wait_any.remote(
                [str(task_id) for task_id in task_ids],
                timeout_seconds,
                max_results,
            ),
            timeout=max(self._rpc_timeout_seconds, timeout_seconds + self._long_poll_margin_seconds),
        )
        data = self._expect_object(response, "WaitAny response")
        snapshots = data.get("tasks")
        if not isinstance(snapshots, list):
            raise AgentServiceProtocolError("WaitAny response.tasks must be a JSON array")
        return [TaskSnapshot.from_dict(self._expect_object(item, "WaitAny task snapshot")) for item in snapshots]

    def cancel(self, task_id: TaskId) -> TaskSnapshot:
        self._ensure_open()
        response = ray.get(
            self._actor.cancel.remote(str(task_id)),
            timeout=self._rpc_timeout_seconds,
        )
        return TaskSnapshot.from_dict(self._expect_object(response, "CancelTask response"))

    def close(self) -> None:
        self._closed = True

    @staticmethod
    def _expect_object(value: Any, context: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise AgentServiceProtocolError(f"{context} must be a JSON object")
        return value

    def _ensure_open(self) -> None:
        if self._closed:
            raise AgentServiceClosedError("RayTransportClient is closed")
