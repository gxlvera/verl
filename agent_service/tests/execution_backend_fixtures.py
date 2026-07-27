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

"""Importable coroutine entrypoints used by spawned Backend workers."""

import asyncio
import os

from agent_service.execution_backend import CoroutineLaunchSpec, TaskExecutionSpec


async def echo_agent(spec: TaskExecutionSpec):
    """Echo the configured payload together with the persistent worker PID."""

    assert isinstance(spec.runtime, CoroutineLaunchSpec)
    await asyncio.sleep(float(spec.runtime.payload.get("delay", 0)))
    return {
        "pid": os.getpid(),
        "value": spec.runtime.payload.get("value"),
    }


async def failing_agent(spec: TaskExecutionSpec):
    """Raise a deterministic exception for failure propagation tests."""

    raise RuntimeError("fixture failure")


class ClassAgent:
    """Zero-argument class entrypoint implementing the coroutine contract."""

    async def run(self, spec: TaskExecutionSpec):
        assert isinstance(spec.runtime, CoroutineLaunchSpec)
        return {"class_value": spec.runtime.payload.get("value")}
