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

import asyncio
import sys

import pytest

from agent_service.execution_backend import (
    CommandExecutionSpec,
    CoroutineLaunchSpec,
    CoroutineRuntimeConfig,
    ExecutionState,
    LocalEnvironmentSpec,
    LocalExecutionBackend,
    LocalExecutionBackendConfig,
    ProcessLaunchSpec,
    ProcessRuntimeConfig,
    SessionCleanedUpError,
    SessionLaunchConflictError,
    TaskExecutionSpec,
    create_execution_backend,
)

FIXTURES = "agent_service.tests.execution_backend_fixtures"


def _coroutine_spec(*, value="ok", delay=0.0, timeout_seconds=5.0, environment=None):
    return TaskExecutionSpec(
        runtime=CoroutineLaunchSpec(
            entrypoint=f"{FIXTURES}:echo_agent",
            payload={"value": value, "delay": delay},
        ),
        environment=environment or LocalEnvironmentSpec(),
        timeout_seconds=timeout_seconds,
    )


def _process_spec(*command, timeout_seconds=5.0, environment=None, stdin=None):
    return TaskExecutionSpec(
        runtime=ProcessLaunchSpec(command=command, stdin=stdin),
        environment=environment or LocalEnvironmentSpec(),
        timeout_seconds=timeout_seconds,
    )


def test_local_backend_config_is_a_strict_discriminated_union():
    coroutine = LocalExecutionBackendConfig.from_dict(
        {
            "kind": "local",
            "runtime": {"kind": "coroutine", "worker_processes": 4},
        }
    )
    process = LocalExecutionBackendConfig.from_dict(
        {
            "kind": "local",
            "runtime": {"kind": "process"},
        }
    )

    assert coroutine.runtime == CoroutineRuntimeConfig(worker_processes=4)
    assert process.runtime == ProcessRuntimeConfig()
    assert isinstance(
        create_execution_backend({"kind": "local", "runtime": {"kind": "coroutine", "worker_processes": 4}}),
        LocalExecutionBackend,
    )

    with pytest.raises(ValueError, match="worker_processes"):
        LocalExecutionBackendConfig.from_dict({"kind": "local", "runtime": {"kind": "coroutine"}})
    with pytest.raises(ValueError, match="Unknown process runtime fields"):
        LocalExecutionBackendConfig.from_dict({"kind": "local", "runtime": {"kind": "process", "worker_processes": 4}})
    with pytest.raises(ValueError, match="Unknown local execution backend fields"):
        LocalExecutionBackendConfig.from_dict(
            {"kind": "local", "runtime": {"kind": "process"}, "max_concurrent_tasks": 512}
        )


@pytest.mark.asyncio
async def test_coroutine_pool_balances_sessions_and_preserves_environment(tmp_path):
    backend = LocalExecutionBackend(
        LocalExecutionBackendConfig(
            runtime=CoroutineRuntimeConfig(worker_processes=2),
            terminate_grace_seconds=0.2,
        )
    )
    await backend.start()
    try:
        session_ids = [f"session-{index}" for index in range(4)]
        for index, session_id in enumerate(session_ids):
            ack = await backend.launch(session_id, _coroutine_spec(value=index, delay=0.2))
            assert ack.created

        results = await asyncio.gather(*(backend.wait(item) for item in session_ids))
        assert {result.status for result in results} == {ExecutionState.SUCCEEDED}
        assert {result.output["value"] for result in results} == {0, 1, 2, 3}
        assert len({result.output["pid"] for result in results}) == 2

        environment = LocalEnvironmentSpec(
            working_dir=str(tmp_path),
            variables={"BACKEND_FIXTURE": "visible"},
        )
        await backend.launch("environment", _coroutine_spec(environment=environment))
        await backend.wait("environment")
        command = await backend.execute_in_environment(
            "environment",
            CommandExecutionSpec(
                command=(
                    sys.executable,
                    "-c",
                    "import os; print(os.getcwd()); print(os.environ['BACKEND_FIXTURE'])",
                )
            ),
        )
        assert command.status is ExecutionState.SUCCEEDED
        assert command.stdout.splitlines() == [str(tmp_path), "visible"]
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_coroutine_launch_is_idempotent_and_supports_cancel_timeout_and_cleanup():
    backend = LocalExecutionBackend(
        LocalExecutionBackendConfig(
            runtime=CoroutineRuntimeConfig(worker_processes=1),
            terminate_grace_seconds=0.2,
        )
    )
    await backend.start()
    try:
        spec = _coroutine_spec(value="same")
        assert (await backend.launch("idempotent", spec)).created
        assert not (await backend.launch("idempotent", spec)).created
        with pytest.raises(SessionLaunchConflictError):
            await backend.launch("idempotent", _coroutine_spec(value="different"))
        assert (await backend.wait("idempotent")).status is ExecutionState.SUCCEEDED

        await backend.launch("cancelled", _coroutine_spec(delay=30, timeout_seconds=None))
        await backend.cancel("cancelled")
        assert (await backend.wait("cancelled")).status is ExecutionState.CANCELLED

        await backend.launch("timed-out", _coroutine_spec(delay=30, timeout_seconds=0.05))
        assert (await backend.wait("timed-out")).status is ExecutionState.TIMED_OUT

        class_spec = TaskExecutionSpec(
            runtime=CoroutineLaunchSpec(entrypoint=f"{FIXTURES}:ClassAgent", payload={"value": 7})
        )
        await backend.launch("class-agent", class_spec)
        assert (await backend.wait("class-agent")).output == {"class_value": 7}

        failure_spec = TaskExecutionSpec(
            runtime=CoroutineLaunchSpec(entrypoint=f"{FIXTURES}:failing_agent", payload={})
        )
        await backend.launch("failed", failure_spec)
        failure = await backend.wait("failed")
        assert failure.status is ExecutionState.FAILED
        assert failure.error is not None
        assert failure.error.code == "AGENT_EXCEPTION"

        await backend.cleanup("idempotent")
        await backend.cleanup("idempotent")
        with pytest.raises(SessionCleanedUpError):
            await backend.inspect("idempotent")
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_process_runtime_reports_output_failure_timeout_cancel_and_environment(tmp_path):
    backend = LocalExecutionBackend(
        LocalExecutionBackendConfig(
            runtime=ProcessRuntimeConfig(),
            terminate_grace_seconds=0.2,
        )
    )
    await backend.start()
    try:
        environment = LocalEnvironmentSpec(
            working_dir=str(tmp_path),
            variables={"BACKEND_FIXTURE": "process-visible"},
        )
        await backend.launch(
            "success",
            _process_spec(
                sys.executable,
                "-c",
                "import os,sys; print(os.getcwd()); print(os.environ['BACKEND_FIXTURE']); print(sys.stdin.read())",
                environment=environment,
                stdin="payload",
            ),
        )
        success = await backend.wait("success")
        assert success.status is ExecutionState.SUCCEEDED
        assert success.exit_code == 0
        assert success.stdout.splitlines() == [str(tmp_path), "process-visible", "payload"]

        verifier = await backend.execute_in_environment(
            "success",
            CommandExecutionSpec(command=(sys.executable, "-c", "print('verified')")),
        )
        assert verifier.status is ExecutionState.SUCCEEDED
        assert verifier.stdout == "verified\n"

        await backend.launch("failed", _process_spec(sys.executable, "-c", "raise SystemExit(7)"))
        failed = await backend.wait("failed")
        assert failed.status is ExecutionState.FAILED
        assert failed.exit_code == 7

        await backend.launch(
            "timed-out",
            _process_spec(sys.executable, "-c", "import time; time.sleep(30)", timeout_seconds=0.05),
        )
        assert (await backend.wait("timed-out")).status is ExecutionState.TIMED_OUT

        await backend.launch(
            "cancelled",
            _process_spec(sys.executable, "-c", "import time; time.sleep(30)", timeout_seconds=None),
        )
        await backend.cancel("cancelled")
        assert (await backend.wait("cancelled")).status is ExecutionState.CANCELLED
    finally:
        await backend.close()
