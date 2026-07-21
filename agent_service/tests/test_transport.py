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

import sys

import pytest

from agent_service import AgentServiceClosedError, RayTransportClient, TaskId, TaskSpec, TaskStatus


class _RemoteMethod:
    def __init__(self, method):
        self._method = method

    def remote(self, *args):
        return self._method(*args)


class _ActorHandle:
    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self.actor, name))


class _ServiceActor:
    def __init__(self):
        self.calls = []

    def submit(self, task_spec, idempotency_key):
        self.calls.append(("submit", task_spec, idempotency_key))
        return {"task_id": "task-1"}

    def get_status(self, task_id):
        self.calls.append(("get_status", task_id))
        return {"task_id": task_id, "status": "RUNNING"}

    def wait_any(self, task_ids, timeout_seconds, max_results):
        self.calls.append(("wait_any", task_ids, timeout_seconds, max_results))
        return {"tasks": [{"task_id": task_ids[0], "status": "SUCCEEDED", "reward": 1.0}]}

    def cancel(self, task_id):
        self.calls.append(("cancel", task_id))
        return {"task_id": task_id, "status": "CANCELLED"}


class _FakeRay:
    class exceptions:
        class GetTimeoutError(Exception):
            pass

    def __init__(self, actor):
        self.actor_handle = _ActorHandle(actor)
        self.lookups = []
        self.timeouts = []

    def get_actor(self, name, *, namespace):
        self.lookups.append((name, namespace))
        return self.actor_handle

    def get(self, value, *, timeout):
        self.timeouts.append(timeout)
        return value


def _task_spec():
    return TaskSpec(problem={}, agent={}, execution={}, reward={}, generation={})


def test_ray_transport_implements_task_api_over_named_actor(monkeypatch):
    actor = _ServiceActor()
    ray = _FakeRay(actor)
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr("agent_service.client_sdk.transport.ray", ray)
    endpoint = {"transport": "ray", "actor_name": "agent-service-1", "namespace": "training"}
    transport_client = RayTransportClient(endpoint)

    task_id = transport_client.submit(_task_spec(), idempotency_key="sample-123")
    running = transport_client.get_status(task_id)
    completed = transport_client.wait_any([task_id], timeout_seconds=2, max_results=1)
    cancelled = transport_client.cancel(task_id)

    assert task_id == TaskId("task-1")
    assert running.status is TaskStatus.RUNNING
    assert completed[0].status is TaskStatus.SUCCEEDED
    assert cancelled.status is TaskStatus.CANCELLED
    assert ray.lookups == [("agent-service-1", "training")]
    assert actor.calls == [
        ("submit", _task_spec().to_dict(), "sample-123"),
        ("get_status", "task-1"),
        ("wait_any", ["task-1"], 2, 1),
        ("cancel", "task-1"),
    ]
    assert all(not isinstance(argument, _ActorHandle) for call in actor.calls for argument in call[1:])


def test_ray_transport_close_is_local_and_rejects_more_rpcs(monkeypatch):
    ray = _FakeRay(_ServiceActor())
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.setattr("agent_service.client_sdk.transport.ray", ray)
    transport_client = RayTransportClient({"transport": "ray", "actor_name": "agent-service-1"})

    transport_client.close()

    with pytest.raises(AgentServiceClosedError):
        transport_client.get_status(TaskId("task-1"))


def test_ray_transport_endpoint_rejects_an_embedded_actor_handle():
    with pytest.raises(ValueError, match="Unknown Agent Service endpoint fields"):
        RayTransportClient(
            {
                "transport": "ray",
                "actor_name": "agent-service-1",
                "actor_handle": _ActorHandle(_ServiceActor()),
            }
        )
