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

"""LocalExecutionBackend with coroutine-worker-pool and per-Task process modes."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import multiprocessing
import os
import signal
import traceback as traceback_module
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from .base import ExecutionBackend
from .errors import (
    BackendNotRunningError,
    ExecutionStillRunningError,
    SessionCleanedUpError,
    SessionLaunchConflictError,
    UnknownSessionError,
    WorkerProcessError,
)
from .models import (
    CommandExecutionSpec,
    CommandResult,
    CoroutineLaunchSpec,
    ExecutionError,
    ExecutionResult,
    ExecutionSnapshot,
    ExecutionState,
    LaunchAck,
    LocalEnvironmentSpec,
    LocalRuntimeKind,
    ProcessLaunchSpec,
    TaskExecutionSpec,
    _json_value,
)


@dataclass(frozen=True)
class CoroutineRuntimeConfig:
    """Fixed local worker pool; every worker hosts multiple AgentLoop coroutines."""

    worker_processes: int

    def __post_init__(self) -> None:
        if not isinstance(self.worker_processes, int) or isinstance(self.worker_processes, bool):
            raise TypeError("execution_backend.runtime.worker_processes must be an integer")
        if self.worker_processes <= 0:
            raise ValueError("execution_backend.runtime.worker_processes must be greater than zero")


@dataclass(frozen=True)
class ProcessRuntimeConfig:
    """One local subprocess per black-box Agent Task."""


LocalRuntimeConfig = CoroutineRuntimeConfig | ProcessRuntimeConfig


@dataclass(frozen=True)
class LocalExecutionBackendConfig:
    """Experiment-scoped configuration for LocalExecutionBackend."""

    runtime: LocalRuntimeConfig
    terminate_grace_seconds: float = 10.0
    worker_start_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, CoroutineRuntimeConfig | ProcessRuntimeConfig):
            raise TypeError("runtime must be a CoroutineRuntimeConfig or ProcessRuntimeConfig")
        for name in ("terminate_grace_seconds", "worker_start_timeout_seconds"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"execution_backend.{name} must be greater than zero")

    @property
    def runtime_kind(self) -> LocalRuntimeKind:
        if isinstance(self.runtime, CoroutineRuntimeConfig):
            return LocalRuntimeKind.COROUTINE
        return LocalRuntimeKind.PROCESS

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LocalExecutionBackendConfig:
        if not isinstance(value, Mapping):
            raise TypeError("execution_backend must be a mapping")
        if value.get("kind") != "local":
            raise ValueError("LocalExecutionBackend requires execution_backend.kind='local'")
        unknown_backend_fields = set(value) - {
            "kind",
            "runtime",
            "terminate_grace_seconds",
            "worker_start_timeout_seconds",
        }
        if unknown_backend_fields:
            raise ValueError(f"Unknown local execution backend fields: {sorted(unknown_backend_fields)}")
        runtime_value = value.get("runtime")
        if not isinstance(runtime_value, Mapping):
            raise ValueError("execution_backend.runtime must be a mapping")
        runtime_kind = runtime_value.get("kind")
        if runtime_kind == LocalRuntimeKind.COROUTINE.value:
            unknown = set(runtime_value) - {"kind", "worker_processes"}
            if unknown:
                raise ValueError(f"Unknown coroutine runtime fields: {sorted(unknown)}")
            if "worker_processes" not in runtime_value:
                raise ValueError("execution_backend.runtime.worker_processes is required for coroutine runtime")
            runtime = CoroutineRuntimeConfig(worker_processes=runtime_value.get("worker_processes"))
        elif runtime_kind == LocalRuntimeKind.PROCESS.value:
            unknown = set(runtime_value) - {"kind"}
            if unknown:
                raise ValueError(f"Unknown process runtime fields: {sorted(unknown)}")
            runtime = ProcessRuntimeConfig()
        else:
            raise ValueError("execution_backend.runtime.kind must be 'coroutine' or 'process'")
        return cls(
            runtime=runtime,
            terminate_grace_seconds=value.get("terminate_grace_seconds", 10.0),
            worker_start_timeout_seconds=value.get("worker_start_timeout_seconds", 30.0),
        )


@dataclass
class _ExecutionRecord:
    session_id: str
    spec: TaskExecutionSpec
    fingerprint: str
    state: ExecutionState
    result_future: asyncio.Future[ExecutionResult]
    worker_id: int | None = None
    process: asyncio.subprocess.Process | None = None
    runner_task: asyncio.Task[None] | None = None
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _WorkerSlot:
    worker_id: int
    process: multiprocessing.Process
    connection: Connection
    send_lock: asyncio.Lock
    ready_future: asyncio.Future[None]
    listener_task: asyncio.Task[None] | None = None
    sessions: set[str] = field(default_factory=set)
    alive: bool = True


@dataclass
class _ChildExecutionRecord:
    spec: TaskExecutionSpec
    task: asyncio.Task[ExecutionResult]
    result: ExecutionResult | None = None


def _load_entrypoint(path: str) -> Any:
    if ":" in path:
        module_name, attribute_path = path.split(":", 1)
    elif "." in path:
        module_name, attribute_path = path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid coroutine entrypoint {path!r}; expected 'module:attribute'")
    loaded = importlib.import_module(module_name)
    for attribute in attribute_path.split("."):
        loaded = getattr(loaded, attribute)
    return loaded


async def _invoke_coroutine_entrypoint(spec: TaskExecutionSpec) -> Any:
    runtime = spec.runtime
    if not isinstance(runtime, CoroutineLaunchSpec):
        raise TypeError("Coroutine worker received a non-coroutine TaskExecutionSpec")
    entrypoint = _load_entrypoint(runtime.entrypoint)
    if isinstance(entrypoint, type):
        instance = entrypoint()
        callable_entrypoint = getattr(instance, "run", None)
        if callable_entrypoint is None:
            raise TypeError(f"Coroutine entrypoint class {runtime.entrypoint!r} has no run method")
    else:
        callable_entrypoint = entrypoint
    if not callable(callable_entrypoint):
        raise TypeError(f"Coroutine entrypoint {runtime.entrypoint!r} is not callable")
    result = callable_entrypoint(spec)
    if not inspect.isawaitable(result):
        raise TypeError(f"Coroutine entrypoint {runtime.entrypoint!r} did not return an awaitable")
    return _json_value(await result, "coroutine result")


def _environment_variables(
    environment: LocalEnvironmentSpec,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    variables = dict(os.environ) if environment.inherit_parent_variables else {}
    variables.update(environment.variables)
    variables.update(overrides or {})
    return variables


def _command_working_dir(environment: LocalEnvironmentSpec, command_working_dir: str | None = None) -> str | None:
    base = Path(environment.working_dir).resolve() if environment.working_dir is not None else None
    if command_working_dir is None:
        return str(base) if base is not None else None
    requested = Path(command_working_dir)
    if requested.is_absolute() or base is None:
        return str(requested.resolve())
    return str((base / requested).resolve())


async def _terminate_process(process: asyncio.subprocess.Process, grace_seconds: float) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    await process.wait()


async def _communicate_with_status(
    process: asyncio.subprocess.Process,
    *,
    stdin: str | None,
    timeout_seconds: float | None,
    terminate_grace_seconds: float,
) -> tuple[ExecutionState, bytes, bytes]:
    communicate_task = asyncio.create_task(process.communicate(None if stdin is None else stdin.encode()))
    try:
        if timeout_seconds is None:
            stdout, stderr = await communicate_task
        else:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=timeout_seconds)
        status = ExecutionState.SUCCEEDED if process.returncode == 0 else ExecutionState.FAILED
        return status, stdout, stderr
    except TimeoutError:
        await _terminate_process(process, terminate_grace_seconds)
        stdout, stderr = await communicate_task
        return ExecutionState.TIMED_OUT, stdout, stderr
    except asyncio.CancelledError:
        await _terminate_process(process, terminate_grace_seconds)
        stdout, stderr = await communicate_task
        return ExecutionState.CANCELLED, stdout, stderr


async def _run_local_command(
    command: CommandExecutionSpec,
    environment: LocalEnvironmentSpec,
    terminate_grace_seconds: float,
) -> CommandResult:
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            *command.command,
            cwd=_command_working_dir(environment, command.working_dir),
            env=_environment_variables(environment, command.environment_variables),
            stdin=asyncio.subprocess.PIPE if command.stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        status, stdout, stderr = await _communicate_with_status(
            process,
            stdin=command.stdin,
            timeout_seconds=command.timeout_seconds,
            terminate_grace_seconds=terminate_grace_seconds,
        )
        error = None
        if status is ExecutionState.FAILED:
            error = ExecutionError(code="COMMAND_EXIT", message=f"Command exited with code {process.returncode}")
        elif status is ExecutionState.TIMED_OUT:
            error = ExecutionError(code="COMMAND_TIMEOUT", message="Command exceeded its timeout")
        elif status is ExecutionState.CANCELLED:
            error = ExecutionError(code="COMMAND_CANCELLED", message="Command was cancelled")
        return CommandResult(
            status=status,
            exit_code=process.returncode,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            error=error,
        )
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_process(process, terminate_grace_seconds)
        raise
    except BaseException as exc:
        return CommandResult(
            status=ExecutionState.FAILED,
            exit_code=process.returncode if process is not None else None,
            error=ExecutionError(
                code="COMMAND_LAUNCH_FAILED",
                message=str(exc),
                traceback=traceback_module.format_exc(),
            ),
        )


async def _child_send(connection: Connection, lock: asyncio.Lock, event: Mapping[str, Any]) -> None:
    async with lock:
        await asyncio.to_thread(connection.send, dict(event))


async def _run_child_coroutine(
    *,
    session_id: str,
    spec: TaskExecutionSpec,
    connection: Connection,
    send_lock: asyncio.Lock,
) -> ExecutionResult:
    await _child_send(connection, send_lock, {"type": "started", "session_id": session_id})
    try:
        if spec.timeout_seconds is None:
            output = await _invoke_coroutine_entrypoint(spec)
        else:
            output = await asyncio.wait_for(_invoke_coroutine_entrypoint(spec), timeout=spec.timeout_seconds)
        result = ExecutionResult(status=ExecutionState.SUCCEEDED, output=output)
    except TimeoutError:
        result = ExecutionResult(
            status=ExecutionState.TIMED_OUT,
            error=ExecutionError(code="AGENT_TIMEOUT", message="AgentLoop exceeded its timeout"),
        )
    except asyncio.CancelledError:
        result = ExecutionResult(
            status=ExecutionState.CANCELLED,
            error=ExecutionError(code="AGENT_CANCELLED", message="AgentLoop was cancelled"),
        )
    except BaseException as exc:
        result = ExecutionResult(
            status=ExecutionState.FAILED,
            error=ExecutionError(
                code="AGENT_EXCEPTION",
                message=str(exc),
                traceback=traceback_module.format_exc(),
            ),
        )
    await _child_send(
        connection,
        send_lock,
        {"type": "completed", "session_id": session_id, "result": result},
    )
    return result


async def _coroutine_worker_loop(connection: Connection, terminate_grace_seconds: float) -> None:
    send_lock = asyncio.Lock()
    records: dict[str, _ChildExecutionRecord] = {}
    control_tasks: set[asyncio.Task[None]] = set()
    await _child_send(connection, send_lock, {"type": "ready"})

    async def respond(request_id: str, *, value: Any = None, error: str | None = None) -> None:
        await _child_send(
            connection,
            send_lock,
            {"type": "response", "request_id": request_id, "value": value, "error": error},
        )

    async def cancel_session(request_id: str, session_id: str) -> None:
        try:
            record = records[session_id]
            if not record.task.done():
                record.task.cancel()
            try:
                record.result = await record.task
            except asyncio.CancelledError:
                # The coroutine may be cancelled before its body has a chance
                # to translate cancellation into an ExecutionResult.
                record.result = ExecutionResult(
                    status=ExecutionState.CANCELLED,
                    error=ExecutionError(code="AGENT_CANCELLED", message="AgentLoop was cancelled"),
                )
                await _child_send(
                    connection,
                    send_lock,
                    {"type": "completed", "session_id": session_id, "result": record.result},
                )
            await respond(request_id)
        except BaseException as exc:
            await respond(request_id, error=f"{type(exc).__name__}: {exc}")

    async def execute_command(request_id: str, session_id: str, command: CommandExecutionSpec) -> None:
        try:
            record = records[session_id]
            if not record.task.done():
                raise ExecutionStillRunningError(f"Session {session_id!r} is still running")
            result = await _run_local_command(command, record.spec.environment, terminate_grace_seconds)
            await respond(request_id, value=result)
        except BaseException as exc:
            await respond(request_id, error=f"{type(exc).__name__}: {exc}")

    async def cleanup_session(request_id: str, session_id: str) -> None:
        try:
            record = records[session_id]
            if not record.task.done():
                raise ExecutionStillRunningError(f"Session {session_id!r} is still running")
            del records[session_id]
            await respond(request_id)
        except BaseException as exc:
            await respond(request_id, error=f"{type(exc).__name__}: {exc}")

    def track_control_task(task: asyncio.Task[None]) -> None:
        control_tasks.add(task)
        task.add_done_callback(control_tasks.discard)

    while True:
        try:
            message = await asyncio.to_thread(connection.recv)
        except EOFError:
            break
        message_type = message.get("type")
        if message_type == "launch":
            session_id = message["session_id"]
            spec = message["spec"]
            task = asyncio.create_task(
                _run_child_coroutine(
                    session_id=session_id,
                    spec=spec,
                    connection=connection,
                    send_lock=send_lock,
                ),
                name=f"agent-loop:{session_id}",
            )
            records[session_id] = _ChildExecutionRecord(spec=spec, task=task)
        elif message_type == "cancel":
            track_control_task(asyncio.create_task(cancel_session(message["request_id"], message["session_id"])))
        elif message_type == "execute":
            track_control_task(
                asyncio.create_task(execute_command(message["request_id"], message["session_id"], message["command"]))
            )
        elif message_type == "cleanup":
            track_control_task(asyncio.create_task(cleanup_session(message["request_id"], message["session_id"])))
        elif message_type == "shutdown":
            for record in records.values():
                if not record.task.done():
                    record.task.cancel()
            await asyncio.gather(*(record.task for record in records.values()), return_exceptions=True)
            for task in tuple(control_tasks):
                task.cancel()
            await asyncio.gather(*control_tasks, return_exceptions=True)
            await respond(message["request_id"])
            await _child_send(connection, send_lock, {"type": "shutdown_ack"})
            break
        else:
            request_id = message.get("request_id")
            if request_id is not None:
                await respond(request_id, error=f"Unknown worker command: {message_type!r}")


def _coroutine_worker_main(connection: Connection, terminate_grace_seconds: float) -> None:
    try:
        asyncio.run(_coroutine_worker_loop(connection, terminate_grace_seconds))
    finally:
        connection.close()


class LocalExecutionBackend(ExecutionBackend):
    """Local Backend selected as either a coroutine pool or subprocess-per-Task."""

    def __init__(self, config: LocalExecutionBackendConfig):
        self.config = config
        self._records: dict[str, _ExecutionRecord] = {}
        self._retired_sessions: dict[str, str] = {}
        self._workers: list[_WorkerSlot] = []
        self._pending_requests: dict[str, asyncio.Future[Any]] = {}
        self._next_worker = 0
        self._started = False
        self._closed = False

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> LocalExecutionBackend:
        return cls(LocalExecutionBackendConfig.from_dict(value))

    async def start(self) -> None:
        if self._closed:
            raise BackendNotRunningError("LocalExecutionBackend is closed")
        if self._started:
            return
        self._started = True
        if isinstance(self.config.runtime, ProcessRuntimeConfig):
            return

        context = multiprocessing.get_context("spawn")
        loop = asyncio.get_running_loop()
        try:
            for worker_id in range(self.config.runtime.worker_processes):
                parent_connection, child_connection = context.Pipe(duplex=True)
                process = context.Process(
                    target=_coroutine_worker_main,
                    args=(child_connection, self.config.terminate_grace_seconds),
                    name=f"agent-service-local-worker-{worker_id}",
                )
                process.start()
                child_connection.close()
                slot = _WorkerSlot(
                    worker_id=worker_id,
                    process=process,
                    connection=parent_connection,
                    send_lock=asyncio.Lock(),
                    ready_future=loop.create_future(),
                )
                slot.listener_task = asyncio.create_task(
                    self._listen_to_worker(slot),
                    name=f"local-worker-listener:{worker_id}",
                )
                self._workers.append(slot)
            await asyncio.wait_for(
                asyncio.gather(*(worker.ready_future for worker in self._workers)),
                timeout=self.config.worker_start_timeout_seconds,
            )
        except BaseException:
            await self.close()
            raise

    async def launch(self, session_id: str, spec: TaskExecutionSpec) -> LaunchAck:
        self._ensure_running()
        self._validate_session_id(session_id)
        if not isinstance(spec, TaskExecutionSpec):
            raise TypeError("spec must be a TaskExecutionSpec")
        if spec.runtime_kind is not self.config.runtime_kind:
            raise ValueError(
                f"Task runtime {spec.runtime_kind.value!r} does not match LocalExecutionBackend runtime "
                f"{self.config.runtime_kind.value!r}"
            )

        fingerprint = spec.fingerprint()
        if session_id in self._retired_sessions:
            raise SessionCleanedUpError(f"Session {session_id!r} has already been cleaned up")
        existing = self._records.get(session_id)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                raise SessionLaunchConflictError(f"Session {session_id!r} was launched with a different spec")
            return LaunchAck(session_id=session_id, created=False)

        record = _ExecutionRecord(
            session_id=session_id,
            spec=spec,
            fingerprint=fingerprint,
            state=ExecutionState.STARTING,
            result_future=asyncio.get_running_loop().create_future(),
        )
        self._records[session_id] = record
        try:
            if isinstance(self.config.runtime, CoroutineRuntimeConfig):
                worker = self._select_worker()
                record.worker_id = worker.worker_id
                worker.sessions.add(session_id)
                await self._send_worker(worker, {"type": "launch", "session_id": session_id, "spec": spec})
            else:
                record.runner_task = asyncio.create_task(
                    self._run_process_record(record),
                    name=f"agent-process:{session_id}",
                )
        except BaseException:
            self._records.pop(session_id, None)
            if record.worker_id is not None:
                self._workers[record.worker_id].sessions.discard(session_id)
            raise
        return LaunchAck(session_id=session_id, created=True)

    async def inspect(self, session_id: str) -> ExecutionSnapshot:
        record = self._get_record(session_id)
        result = record.result_future.result() if record.result_future.done() else None
        return ExecutionSnapshot(session_id=session_id, state=record.state, result=result)

    async def wait(self, session_id: str) -> ExecutionResult:
        record = self._get_record(session_id)
        return await asyncio.shield(record.result_future)

    async def cancel(self, session_id: str) -> None:
        record = self._get_record(session_id)
        if record.result_future.done():
            return
        if isinstance(self.config.runtime, CoroutineRuntimeConfig):
            worker = self._worker_for_record(record)
            await self._request_worker(worker, "cancel", session_id=session_id)
        else:
            if record.runner_task is not None and not record.runner_task.done():
                record.runner_task.cancel()
                try:
                    await record.runner_task
                except asyncio.CancelledError:
                    # A Task cancelled before its coroutine body starts cannot
                    # publish its own result, so complete the Backend record here.
                    self._finish_record(
                        record,
                        ExecutionResult(
                            status=ExecutionState.CANCELLED,
                            error=ExecutionError(code="AGENT_CANCELLED", message="Agent process was cancelled"),
                        ),
                    )
        await self.wait(session_id)

    async def execute_in_environment(
        self,
        session_id: str,
        command: CommandExecutionSpec,
    ) -> CommandResult:
        record = self._get_record(session_id)
        if not record.result_future.done():
            raise ExecutionStillRunningError(f"Session {session_id!r} is still running")
        async with record.operation_lock:
            if isinstance(self.config.runtime, CoroutineRuntimeConfig):
                worker = self._worker_for_record(record)
                result = await self._request_worker(worker, "execute", session_id=session_id, command=command)
                if not isinstance(result, CommandResult):
                    raise WorkerProcessError("Coroutine worker returned an invalid CommandResult")
                return result
            return await _run_local_command(command, record.spec.environment, self.config.terminate_grace_seconds)

    async def cleanup(self, session_id: str) -> None:
        if session_id in self._retired_sessions:
            return
        record = self._records.get(session_id)
        if record is None:
            raise UnknownSessionError(session_id)
        if not record.result_future.done():
            raise ExecutionStillRunningError(f"Session {session_id!r} is still running")
        async with record.operation_lock:
            if isinstance(self.config.runtime, CoroutineRuntimeConfig):
                worker = self._worker_for_record(record)
                await self._request_worker(worker, "cleanup", session_id=session_id)
                worker.sessions.discard(session_id)
            record.process = None
            record.runner_task = None
            self._records.pop(session_id, None)
            self._retired_sessions[session_id] = record.fingerprint

    async def close(self) -> None:
        if self._closed:
            return

        active = [record.session_id for record in self._records.values() if not record.result_future.done()]
        await asyncio.gather(*(self.cancel(session_id) for session_id in active), return_exceptions=True)

        if isinstance(self.config.runtime, CoroutineRuntimeConfig):
            await asyncio.gather(
                *(self._request_worker(worker, "shutdown") for worker in self._workers if worker.alive),
                return_exceptions=True,
            )
            listeners = [worker.listener_task for worker in self._workers if worker.listener_task is not None]
            if listeners:
                await asyncio.gather(*listeners, return_exceptions=True)
            for worker in self._workers:
                await asyncio.to_thread(worker.process.join, self.config.terminate_grace_seconds)
                if worker.process.is_alive():
                    worker.process.terminate()
                    await asyncio.to_thread(worker.process.join, self.config.terminate_grace_seconds)
                worker.connection.close()

        self._records.clear()
        self._workers.clear()
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(BackendNotRunningError("LocalExecutionBackend closed"))
        self._pending_requests.clear()
        self._started = False
        self._closed = True

    async def _run_process_record(self, record: _ExecutionRecord) -> None:
        runtime = record.spec.runtime
        assert isinstance(runtime, ProcessLaunchSpec)
        try:
            process = await asyncio.create_subprocess_exec(
                *runtime.command,
                cwd=_command_working_dir(record.spec.environment),
                env=_environment_variables(record.spec.environment),
                stdin=asyncio.subprocess.PIPE if runtime.stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            record.process = process
            record.state = ExecutionState.RUNNING
            status, stdout, stderr = await _communicate_with_status(
                process,
                stdin=runtime.stdin,
                timeout_seconds=record.spec.timeout_seconds,
                terminate_grace_seconds=self.config.terminate_grace_seconds,
            )
            error = None
            if status is ExecutionState.FAILED:
                error = ExecutionError(
                    code="AGENT_EXIT",
                    message=f"Agent process exited with code {process.returncode}",
                )
            elif status is ExecutionState.TIMED_OUT:
                error = ExecutionError(code="AGENT_TIMEOUT", message="Agent process exceeded its timeout")
            elif status is ExecutionState.CANCELLED:
                error = ExecutionError(code="AGENT_CANCELLED", message="Agent process was cancelled")
            result = ExecutionResult(
                status=status,
                exit_code=process.returncode,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                error=error,
            )
        except asyncio.CancelledError:
            if record.process is not None:
                await _terminate_process(record.process, self.config.terminate_grace_seconds)
            result = ExecutionResult(
                status=ExecutionState.CANCELLED,
                exit_code=record.process.returncode if record.process is not None else None,
                error=ExecutionError(code="AGENT_CANCELLED", message="Agent process was cancelled"),
            )
        except BaseException as exc:
            result = ExecutionResult(
                status=ExecutionState.FAILED,
                exit_code=record.process.returncode if record.process is not None else None,
                error=ExecutionError(
                    code="AGENT_LAUNCH_FAILED",
                    message=str(exc),
                    traceback=traceback_module.format_exc(),
                ),
            )
        self._finish_record(record, result)

    def _finish_record(self, record: _ExecutionRecord, result: ExecutionResult) -> None:
        record.state = result.status
        if not record.result_future.done():
            record.result_future.set_result(result)

    async def _listen_to_worker(self, worker: _WorkerSlot) -> None:
        try:
            while True:
                event = await asyncio.to_thread(worker.connection.recv)
                event_type = event.get("type")
                if event_type == "ready":
                    if not worker.ready_future.done():
                        worker.ready_future.set_result(None)
                elif event_type == "started":
                    record = self._records.get(event["session_id"])
                    if record is not None and not record.result_future.done():
                        record.state = ExecutionState.RUNNING
                elif event_type == "completed":
                    session_id = event["session_id"]
                    record = self._records.get(session_id)
                    result = event["result"]
                    if record is not None and isinstance(result, ExecutionResult):
                        self._finish_record(record, result)
                    worker.sessions.discard(session_id)
                elif event_type == "response":
                    future = self._pending_requests.pop(event["request_id"], None)
                    if future is not None and not future.done():
                        error = event.get("error")
                        if error is None:
                            future.set_result(event.get("value"))
                        else:
                            future.set_exception(WorkerProcessError(error))
                elif event_type == "shutdown_ack":
                    break
        except (EOFError, OSError) as exc:
            self._fail_worker(worker, f"Worker connection closed: {exc}")
        finally:
            worker.alive = False
            if not worker.ready_future.done():
                worker.ready_future.set_exception(WorkerProcessError(f"Worker {worker.worker_id} exited before ready"))

    def _fail_worker(self, worker: _WorkerSlot, message: str) -> None:
        worker.alive = False
        for session_id in tuple(worker.sessions):
            record = self._records.get(session_id)
            if record is not None and not record.result_future.done():
                self._finish_record(
                    record,
                    ExecutionResult(
                        status=ExecutionState.FAILED,
                        error=ExecutionError(code="WORKER_EXITED", message=message),
                    ),
                )
        worker.sessions.clear()
        for request_id, future in tuple(self._pending_requests.items()):
            if request_id.startswith(f"{worker.worker_id}:"):
                self._pending_requests.pop(request_id, None)
                if not future.done():
                    future.set_exception(WorkerProcessError(message))

    async def _send_worker(self, worker: _WorkerSlot, message: Mapping[str, Any]) -> None:
        if not worker.alive:
            raise WorkerProcessError(f"Coroutine worker {worker.worker_id} is not running")
        async with worker.send_lock:
            await asyncio.to_thread(worker.connection.send, dict(message))

    async def _request_worker(self, worker: _WorkerSlot, operation: str, **payload: Any) -> Any:
        request_id = f"{worker.worker_id}:{uuid.uuid4().hex}"
        future = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future
        try:
            await self._send_worker(
                worker,
                {"type": operation, "request_id": request_id, **payload},
            )
            return await future
        except BaseException:
            self._pending_requests.pop(request_id, None)
            raise

    def _select_worker(self) -> _WorkerSlot:
        live_workers = [worker for worker in self._workers if worker.alive]
        if not live_workers:
            raise WorkerProcessError("No coroutine workers are running")
        minimum_load = min(len(worker.sessions) for worker in live_workers)
        candidates = [worker for worker in live_workers if len(worker.sessions) == minimum_load]
        selected = candidates[self._next_worker % len(candidates)]
        self._next_worker += 1
        return selected

    def _worker_for_record(self, record: _ExecutionRecord) -> _WorkerSlot:
        if record.worker_id is None or record.worker_id >= len(self._workers):
            raise WorkerProcessError(f"Session {record.session_id!r} has no coroutine worker")
        worker = self._workers[record.worker_id]
        if not worker.alive:
            raise WorkerProcessError(f"Coroutine worker {worker.worker_id} is not running")
        return worker

    def _get_record(self, session_id: str) -> _ExecutionRecord:
        self._ensure_running()
        if session_id in self._retired_sessions:
            raise SessionCleanedUpError(f"Session {session_id!r} has already been cleaned up")
        try:
            return self._records[session_id]
        except KeyError as exc:
            raise UnknownSessionError(session_id) from exc

    def _ensure_running(self) -> None:
        if not self._started or self._closed:
            raise BackendNotRunningError("LocalExecutionBackend is not running")

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
