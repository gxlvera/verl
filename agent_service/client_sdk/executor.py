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

"""Stable Driver API over an Agent Service transport."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Iterator, Sequence

from .errors import (
    AgentServiceClosedError,
    AgentServiceProtocolError,
    AgentServiceTimeoutError,
    MaxInFlightTasksError,
)
from .models import TaskId, TaskSnapshot, TaskSpec
from .transport import TransportClient


class AgentExecutor:
    """Driver-side facade implementing the RFC's TaskFuture + as_completed style.

    Wraps a :class:`TransportClient` and tracks which submitted Tasks have not
    yet reached a terminal state. Driver and rollout code depend only on this
    facade; the deployment-specific transport stays private.
    """

    def __init__(
        self,
        transport_client: TransportClient,
        *,
        max_in_flight: int | None = None,
        wait_any_poll_timeout_seconds: float = 30.0,
        wait_any_max_results: int = 64,
    ) -> None:
        if max_in_flight is not None and max_in_flight <= 0:
            raise ValueError("max_in_flight must be greater than zero or null")
        if wait_any_poll_timeout_seconds <= 0:
            raise ValueError("wait_any_poll_timeout_seconds must be greater than zero")
        if wait_any_max_results <= 0:
            raise ValueError("wait_any_max_results must be greater than zero")

        self._transport_client = transport_client
        self._max_in_flight = max_in_flight
        self._wait_any_poll_timeout_seconds = wait_any_poll_timeout_seconds
        self._wait_any_max_results = wait_any_max_results
        self._in_flight: set[TaskId] = set()
        self._closed = False
        self._lock = threading.Lock()

    @property
    def in_flight_task_ids(self) -> tuple[TaskId, ...]:
        with self._lock:
            return tuple(self._in_flight)

    def submit(self, task_spec: TaskSpec, idempotency_key: str | None = None) -> TaskId:
        """Submit one Task and immediately return its server-assigned ID.

        max_in_flight is a hard local cap that raises instead of blocking: the
        V0 driver submits a whole batch before draining it, so blocking here
        would deadlock. Server-side admission remains the primary backpressure.
        """
        with self._lock:
            self._ensure_open()
            if self._max_in_flight is not None and len(self._in_flight) >= self._max_in_flight:
                raise MaxInFlightTasksError(self._max_in_flight)
        task_id = self._transport_client.submit(task_spec, idempotency_key=idempotency_key)
        with self._lock:
            self._in_flight.add(task_id)
        return task_id

    def submit_future(self, task_spec: TaskSpec, idempotency_key: str | None = None) -> TaskFuture:
        """Submit one Task and wrap its ID in the optional Future convenience API."""

        return TaskFuture(self, self.submit(task_spec, idempotency_key=idempotency_key))

    def get_status(self, task_id: TaskId) -> TaskSnapshot:
        self._ensure_open()
        snapshot = self._transport_client.get_status(task_id)
        self._validate_requested_snapshot(snapshot, {task_id})
        self._mark_terminal(snapshot)
        return snapshot

    def wait_any(
        self,
        task_ids: Sequence[TaskId],
        timeout_seconds: float,
        max_results: int,
    ) -> list[TaskSnapshot]:
        self._ensure_open()
        requested = set(task_ids)
        if not requested:
            return []
        snapshots = self._transport_client.wait_any(tuple(requested), timeout_seconds, max_results)
        # Guard the WaitAny contract (only requested, terminal, distinct Tasks)
        # so a buggy server fails loudly instead of corrupting batch bookkeeping.
        seen: set[TaskId] = set()
        for snapshot in snapshots:
            self._validate_requested_snapshot(snapshot, requested)
            if not snapshot.is_terminal:
                raise AgentServiceProtocolError(
                    f"WaitAny returned non-terminal Task {snapshot.task_id} in state {snapshot.status.value}"
                )
            if snapshot.task_id in seen:
                raise AgentServiceProtocolError(f"WaitAny returned duplicate Task {snapshot.task_id}")
            seen.add(snapshot.task_id)
            self._mark_terminal(snapshot)
        return snapshots

    def cancel(self, task_id: TaskId) -> TaskSnapshot:
        self._ensure_open()
        snapshot = self._transport_client.cancel(task_id)
        self._validate_requested_snapshot(snapshot, {task_id})
        self._mark_terminal(snapshot)
        return snapshot

    def as_completed(
        self,
        task_ids: Iterable[TaskId | TaskFuture],
        timeout: float | None = None,
    ) -> Iterator[TaskSnapshot]:
        return as_completed(self, task_ids, timeout=timeout)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._transport_client.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AgentServiceClosedError("AgentExecutor is closed")

    def _mark_terminal(self, snapshot: TaskSnapshot) -> None:
        if snapshot.is_terminal:
            with self._lock:
                self._in_flight.discard(snapshot.task_id)

    @staticmethod
    def _validate_requested_snapshot(snapshot: TaskSnapshot, requested: set[TaskId]) -> None:
        if snapshot.task_id not in requested:
            raise AgentServiceProtocolError(f"Service returned unrequested Task {snapshot.task_id}")

    def __enter__(self) -> AgentExecutor:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class TaskFuture:
    """A lightweight reference to one remotely executing Agent Service Task."""

    def __init__(self, executor: AgentExecutor, task_id: TaskId):
        self._executor = executor
        self._task_id = task_id
        self._terminal_snapshot: TaskSnapshot | None = None

    @property
    def task_id(self) -> TaskId:
        return self._task_id

    def get_status(self) -> TaskSnapshot:
        if self._terminal_snapshot is not None:
            return self._terminal_snapshot
        snapshot = self._executor.get_status(self._task_id)
        if snapshot.is_terminal:
            self._terminal_snapshot = snapshot
        return snapshot

    def done(self) -> bool:
        return self.get_status().is_terminal

    def cancel(self) -> TaskSnapshot:
        snapshot = self._executor.cancel(self._task_id)
        if snapshot.is_terminal:
            self._terminal_snapshot = snapshot
        return snapshot

    def result(self, timeout: float | None = None) -> TaskSnapshot:
        if self._terminal_snapshot is not None:
            return self._terminal_snapshot
        try:
            snapshot = next(as_completed(self._executor, [self], timeout=timeout))
        except StopIteration as exc:  # Defensive: a pending Task must yield or time out.
            raise AgentServiceProtocolError(f"Task {self._task_id} did not produce a terminal snapshot") from exc
        self._terminal_snapshot = snapshot
        return snapshot


def as_completed(
    executor: AgentExecutor,
    task_ids: Iterable[TaskId | TaskFuture],
    timeout: float | None = None,
) -> Iterator[TaskSnapshot]:
    """Yield terminal snapshots in service completion order using WaitAny long-poll."""

    if timeout is not None and timeout < 0:
        raise ValueError("timeout must not be negative")

    pending: dict[TaskId, TaskFuture | None] = {}
    for item in task_ids:
        if isinstance(item, TaskFuture):
            if item._executor is not executor:
                raise ValueError("All TaskFuture objects must belong to the supplied executor")
            pending[item.task_id] = item
        else:
            pending[TaskId(str(item))] = None
    if not pending:
        return

    deadline = None if timeout is None else time.monotonic() + timeout
    while pending:
        if deadline is None:
            poll_timeout = executor._wait_any_poll_timeout_seconds
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentServiceTimeoutError(
                    f"Timed out waiting for {len(pending)} Agent Service Task(s): {sorted(map(str, pending))}"
                )
            poll_timeout = min(executor._wait_any_poll_timeout_seconds, remaining)

        # An empty result just means the long-poll window elapsed; re-poll
        # until every pending Task completes or the overall deadline expires.
        snapshots = executor.wait_any(
            tuple(pending),
            timeout_seconds=poll_timeout,
            max_results=min(executor._wait_any_max_results, len(pending)),
        )
        for snapshot in snapshots:
            future = pending.pop(snapshot.task_id)
            if future is not None:
                future._terminal_snapshot = snapshot
            yield snapshot
