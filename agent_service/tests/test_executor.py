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

from collections import deque

import pytest

from agent_service import (
    AgentExecutor,
    AgentServiceClosedError,
    AgentServiceProtocolError,
    MaxInFlightTasksError,
    TaskId,
    TaskSnapshot,
    TaskSpec,
    TaskStatus,
)


class _FakeTransportClient:
    """In-memory implementation of the TransportClient protocol."""

    def __init__(self):
        self.next_id = 1
        self.statuses = {}
        self.wait_responses = deque()
        self.closed = False

    def submit(self, task_spec, idempotency_key=None):
        task_id = TaskId(f"task-{self.next_id}")
        self.next_id += 1
        self.statuses[task_id] = TaskSnapshot(task_id=task_id, status=TaskStatus.RUNNING)
        return task_id

    def get_status(self, task_id):
        return self.statuses[task_id]

    def wait_any(self, task_ids, timeout_seconds, max_results):
        if not self.wait_responses:
            return []
        return self.wait_responses.popleft()[:max_results]

    def cancel(self, task_id):
        snapshot = TaskSnapshot(task_id=task_id, status=TaskStatus.CANCELLED)
        self.statuses[task_id] = snapshot
        return snapshot

    def close(self):
        self.closed = True


def _task_spec():
    return TaskSpec(problem={}, agent={}, execution={}, reward={}, generation={})


def _executor(max_in_flight=None):
    transport_client = _FakeTransportClient()
    executor = AgentExecutor(
        transport_client,
        max_in_flight=max_in_flight,
        wait_any_poll_timeout_seconds=1,
    )
    return executor, transport_client


def test_as_completed_yields_service_completion_order_and_releases_slots():
    executor, transport_client = _executor(max_in_flight=2)
    first = executor.submit(_task_spec())
    second = executor.submit(_task_spec())
    transport_client.wait_responses.extend(
        [
            [TaskSnapshot(task_id=second, status=TaskStatus.SUCCEEDED, reward=2.0)],
            [TaskSnapshot(task_id=first, status=TaskStatus.FAILED)],
        ]
    )

    snapshots = list(executor.as_completed([first, second], timeout=5))

    assert [snapshot.task_id for snapshot in snapshots] == [second, first]
    assert snapshots[0].final_reward == 2.0
    assert snapshots[1].status is TaskStatus.FAILED
    assert executor.in_flight_task_ids == ()

    third = executor.submit(_task_spec())
    assert third == TaskId("task-3")


def test_submit_future_is_an_optional_convenience_wrapper():
    executor, transport_client = _executor()
    future = executor.submit_future(_task_spec())
    transport_client.wait_responses.append(
        [TaskSnapshot(task_id=future.task_id, status=TaskStatus.SUCCEEDED, reward=1.0)]
    )

    assert future.result(timeout=5).final_reward == 1.0


def test_max_in_flight_is_enforced_until_terminal_snapshot_is_observed():
    executor, transport_client = _executor(max_in_flight=1)
    task_id = executor.submit(_task_spec())

    with pytest.raises(MaxInFlightTasksError):
        executor.submit(_task_spec())

    transport_client.statuses[task_id] = TaskSnapshot(task_id=task_id, status=TaskStatus.SUCCEEDED)
    assert executor.get_status(task_id).is_terminal
    executor.submit(_task_spec())


def test_wait_any_rejects_non_terminal_or_unrequested_snapshots():
    executor, transport_client = _executor()
    task_id = executor.submit(_task_spec())
    transport_client.wait_responses.append([TaskSnapshot(task_id=task_id, status=TaskStatus.RUNNING)])

    with pytest.raises(AgentServiceProtocolError, match="non-terminal"):
        executor.wait_any([task_id], timeout_seconds=1, max_results=1)

    transport_client.wait_responses.append([TaskSnapshot(task_id=TaskId("other"), status=TaskStatus.SUCCEEDED)])
    with pytest.raises(AgentServiceProtocolError, match="unrequested"):
        executor.wait_any([task_id], timeout_seconds=1, max_results=1)


def test_close_is_idempotent_and_rejects_further_use():
    executor, transport_client = _executor()
    task_id = executor.submit(_task_spec())
    executor.cancel(task_id)

    executor.close()
    executor.close()

    assert transport_client.statuses[task_id].status is TaskStatus.CANCELLED
    assert transport_client.closed
    with pytest.raises(AgentServiceClosedError):
        executor.submit(_task_spec())
