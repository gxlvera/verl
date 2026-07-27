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
from omegaconf import OmegaConf

from agent_service import AgentServiceStartupConfig, InferenceSpec
from agent_service.verl_adapter.runtime import VerlAgentServiceRuntime, _build_proxy_config, _default_actor_num_cpus


class _RemoteMethod:
    def __init__(self, method):
        self._method = method

    def remote(self, *args, **kwargs):
        return self._method(*args, **kwargs)


class _ActorHandle:
    def __init__(self, actor):
        self.actor = actor

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self.actor, name))


class _ActorClass:
    def __init__(self, actor_class, ray_module):
        self.actor_class = actor_class
        self.ray_module = ray_module

    def options(self, **options):
        self.ray_module.options = options
        return self

    def remote(self, startup_config):
        self.ray_module.startup_config = startup_config
        self.ray_module.actor_handle = _ActorHandle(self.actor_class(startup_config))
        return self.ray_module.actor_handle


class _FakeRay:
    def __init__(self):
        self.options = None
        self.startup_config = None
        self.actor_handle = None
        self.killed = []
        self.get_timeouts = []
        self.lookups = []

    def remote(self, actor_class):
        return _ActorClass(actor_class, self)

    def get(self, value, timeout=None):
        self.get_timeouts.append(timeout)
        return value

    @staticmethod
    def get_runtime_context():
        return type("RuntimeContext", (), {"namespace": "training"})()

    def get_actor(self, name, *, namespace):
        self.lookups.append((name, namespace))
        return self.actor_handle

    def kill(self, actor_handle, *, no_restart):
        self.killed.append((actor_handle, no_restart))


class _ServiceActor:
    def __init__(self, startup_config):
        self.startup_config = startup_config
        self.stopped_with = None

    @staticmethod
    def wait_ready(timeout_seconds):
        assert timeout_seconds == 12

    def stop(self, graceful_timeout_seconds):
        self.stopped_with = graceful_timeout_seconds


class _RolloutAdapter:
    def __init__(self, *, executor, config, tokenizer, processor=None):
        self.executor = executor
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.cancelled = False

    def cancel_in_flight(self):
        self.cancelled = True


class _Trainer:
    @staticmethod
    def get_rollout_inference_addresses():
        return ["http://replica:8000"]


def _config(*, actor_options=None):
    if actor_options is None:
        actor_options = {"num_cpus": 2, "name": "agent-service-test"}
    return OmegaConf.create(
        {
            "agent_service": {
                "ray_actor": {
                    "actor_class": "tests.ServiceActor",
                    "actor_options": actor_options,
                },
                "execution_backend": {
                    "kind": "local",
                    "runtime": {"kind": "coroutine", "worker_processes": 4},
                },
                "upstream_protocol": "openai_chat_completions",
                "proxy": {"replicas": 2},
                "admission": {"max_concurrent_tasks": 8},
                "default_lifecycle": {"timeout_seconds": 300},
                "ready_timeout_seconds": 12,
                "rpc_timeout_seconds": 3,
                "shutdown_timeout_seconds": 5,
            },
            "actor_rollout_ref": {
                "model": {
                    "path": "hdfs://models/policy",
                    "tokenizer_path": None,
                }
            },
        }
    )


def test_proxy_config_carries_resolved_verl_model_config_as_plain_data():
    config = OmegaConf.create(
        {
            "artifact_root": "hdfs://models/policy",
            "actor_rollout_ref": {
                "model": {
                    "_target_": "verl.workers.config.HFModelConfig",
                    "path": "${artifact_root}",
                    "tokenizer_path": None,
                    "trust_remote_code": True,
                    "custom_chat_template": "{{ messages }}",
                    "external_lib": ["custom_tokenizer"],
                }
            },
        }
    )

    proxy_config = _build_proxy_config(
        config,
        {"proxy": {"replicas": 2, "model_config": {"path": "stale"}}},
    )

    assert proxy_config == {
        "replicas": 2,
        "model_config": {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": "hdfs://models/policy",
            "tokenizer_path": None,
            "trust_remote_code": True,
            "custom_chat_template": "{{ messages }}",
            "external_lib": ["custom_tokenizer"],
        },
    }
    assert isinstance(proxy_config["model_config"], dict)


def test_runtime_owns_ray_actor_lifecycle(monkeypatch):
    fake_ray = _FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr("agent_service.verl_adapter.runtime.ray", fake_ray)
    monkeypatch.setattr("agent_service.client_sdk.transport.ray", fake_ray)
    config = _config()
    loaded_classes = {
        "tests.ServiceActor": _ServiceActor,
    }
    monkeypatch.setattr("agent_service.verl_adapter.runtime._load_class", loaded_classes.__getitem__)
    monkeypatch.setattr("agent_service.verl_adapter.runtime.RolloutAdapter", _RolloutAdapter)

    tokenizer = object()
    processor = object()
    runtime = VerlAgentServiceRuntime(
        trainer=_Trainer(),
        config=config,
    )
    runtime.start()
    rollout_adapter = runtime.get_rollout_adapter(
        tokenizer=tokenizer,
        processor=processor,
    )

    assert fake_ray.options == {
        "num_cpus": 2,
        "name": "agent-service-test",
    }
    assert fake_ray.startup_config == {
        "execution_backend": {
            "kind": "local",
            "runtime": {"kind": "coroutine", "worker_processes": 4},
        },
        "inference": {
            "replica_endpoints": ["http://replica:8000"],
            "upstream_protocol": "openai_chat_completions",
        },
        "proxy": {
            "replicas": 2,
            "model_config": {
                "path": "hdfs://models/policy",
                "tokenizer_path": None,
            },
        },
        "admission": {"max_concurrent_tasks": 8},
        "default_lifecycle": {"timeout_seconds": 300},
    }

    assert not hasattr(runtime, "ray_actor")
    assert not hasattr(runtime, "executor")
    assert not hasattr(runtime, "rollout_adapter")
    assert rollout_adapter.executor is runtime._executor
    assert rollout_adapter.tokenizer is tokenizer
    assert rollout_adapter.processor is processor
    assert fake_ray.get_timeouts == [12.0]

    runtime.close()
    assert rollout_adapter.cancelled
    assert fake_ray.actor_handle.actor.stopped_with == 5
    assert fake_ray.killed == [(fake_ray.actor_handle, True)]


def test_runtime_requires_start_before_getting_rollout_adapter():
    runtime = VerlAgentServiceRuntime(trainer=_Trainer(), config=_config())

    with pytest.raises(RuntimeError, match="must be started"):
        runtime.get_rollout_adapter(tokenizer=object())


def test_default_actor_cpu_reservation_tracks_local_runtime_workers():
    inference = InferenceSpec(
        replica_endpoints=["http://replica:8000"],
        upstream_protocol="openai_chat_completions",
    )
    coroutine = AgentServiceStartupConfig(
        execution_backend={
            "kind": "local",
            "runtime": {"kind": "coroutine", "worker_processes": 4},
        },
        inference=inference,
    )
    process = AgentServiceStartupConfig(
        execution_backend={"kind": "local", "runtime": {"kind": "process"}},
        inference=inference,
    )

    assert _default_actor_num_cpus(coroutine) == 4
    assert _default_actor_num_cpus(process) == 1
