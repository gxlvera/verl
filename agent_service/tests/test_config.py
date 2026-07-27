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

import pytest

from agent_service import AgentServiceStartupConfig, InferenceSpec
from agent_service.verl_adapter.config import validate_agent_service_config


def test_startup_config_serializes_fixed_inference_registry():
    config = AgentServiceStartupConfig(
        execution_backend={
            "kind": "local",
            "runtime": {"kind": "coroutine", "worker_processes": 4},
        },
        inference=InferenceSpec(
            replica_endpoints=["http://replica-0:8000", "http://replica-1:8000"],
            upstream_protocol="openai_chat_completions",
        ),
        proxy={"model_config": {"path": "hdfs://models/policy", "tokenizer_path": None}},
        admission={"max_concurrent_tasks": 8},
    )

    assert config.to_dict() == {
        "execution_backend": {
            "kind": "local",
            "runtime": {"kind": "coroutine", "worker_processes": 4},
        },
        "inference": {
            "replica_endpoints": ["http://replica-0:8000", "http://replica-1:8000"],
            "upstream_protocol": "openai_chat_completions",
        },
        "proxy": {"model_config": {"path": "hdfs://models/policy", "tokenizer_path": None}},
        "admission": {"max_concurrent_tasks": 8},
    }


def test_v0_rejects_non_local_backend_and_duplicate_replicas():
    with pytest.raises(ValueError, match="only execution_backend.kind='local'"):
        AgentServiceStartupConfig(
            execution_backend={"kind": "ray"},
            inference=InferenceSpec(
                replica_endpoints=["http://replica:8000"],
                upstream_protocol="openai_responses",
            ),
        )

    with pytest.raises(ValueError, match="duplicates"):
        InferenceSpec(
            replica_endpoints=["http://replica:8000", "http://replica:8000"],
            upstream_protocol="openai_responses",
        )


def test_generate_is_supported_only_as_an_upstream_protocol():
    spec = InferenceSpec(
        replica_endpoints=["http://replica:8000"],
        upstream_protocol="generate",
    )

    assert spec.to_dict()["upstream_protocol"] == "generate"


def test_disabled_integration_requires_no_server_bootstrap():
    validate_agent_service_config({"enabled": False})


def test_builtin_adapter_validates_required_task_bootstrap_fields():
    config = {
        "enabled": True,
        "ray_actor": {"actor_class": "package.AgentServiceActor"},
        "execution_backend": {
            "kind": "local",
            "runtime": {"kind": "process"},
        },
        "upstream_protocol": "openai_chat_completions",
        "admission": {"max_concurrent_tasks": 512, "max_queued_tasks": 1024},
        "ready_timeout_seconds": 10,
        "wait_timeout_seconds": 10,
        "shutdown_timeout_seconds": 10,
        "rpc_timeout_seconds": 10,
        "task": {
            "agent": {"artifact": None, "frontend_protocol": "openai_chat_completions"},
            "execution": {"command": ["python", "main.py"]},
            "reward": {"reward_function": {"kind": "binary"}},
        },
    }

    with pytest.raises(ValueError, match="agent.artifact"):
        validate_agent_service_config(config)

    config["task"]["agent"]["artifact"] = "./agent"
    validate_agent_service_config(config)

    config["upstream_protocol"] = "generate"
    validate_agent_service_config(config)

    config["task"]["agent"]["frontend_protocol"] = "generate"
    with pytest.raises(ValueError, match="frontend_protocol"):
        validate_agent_service_config(config)


def test_builtin_adapter_validates_runtime_specific_agent_launch_fields():
    config = {
        "enabled": True,
        "ray_actor": {"actor_class": "package.AgentServiceActor"},
        "execution_backend": {
            "kind": "local",
            "runtime": {"kind": "coroutine", "worker_processes": 4},
        },
        "upstream_protocol": "openai_chat_completions",
        "admission": {"max_concurrent_tasks": 512, "max_queued_tasks": 1024},
        "ready_timeout_seconds": 10,
        "wait_timeout_seconds": 10,
        "shutdown_timeout_seconds": 10,
        "rpc_timeout_seconds": 10,
        "task": {
            "agent": {
                "artifact": "./agent",
                "entrypoint": None,
                "frontend_protocol": "openai_chat_completions",
            },
            "execution": {"command": None},
            "reward": {"reward_function": {"kind": "binary"}},
        },
    }

    with pytest.raises(ValueError, match="agent.entrypoint"):
        validate_agent_service_config(config)

    config["task"]["agent"]["entrypoint"] = "package.module:AgentLoop"
    validate_agent_service_config(config)

    config["admission"]["max_concurrent_tasks"] = 0
    with pytest.raises(ValueError, match="max_concurrent_tasks"):
        validate_agent_service_config(config)
