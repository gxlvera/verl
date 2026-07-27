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

"""Pluggable execution contract consumed by the future AgentTaskController."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import (
    CommandExecutionSpec,
    CommandResult,
    ExecutionResult,
    ExecutionSnapshot,
    LaunchAck,
    TaskExecutionSpec,
)


class ExecutionBackend(ABC):
    """Experiment-scoped owner of per-session native runtime handles."""

    @abstractmethod
    async def start(self) -> None:
        """Allocate Backend-level resources and become ready for launches."""

    @abstractmethod
    async def launch(self, session_id: str, spec: TaskExecutionSpec) -> LaunchAck:
        """Start one AgentRuntime without waiting for it to finish."""

    @abstractmethod
    async def inspect(self, session_id: str) -> ExecutionSnapshot:
        """Return a non-blocking snapshot of one execution."""

    @abstractmethod
    async def wait(self, session_id: str) -> ExecutionResult:
        """Wait for one execution and return its terminal facts."""

    @abstractmethod
    async def cancel(self, session_id: str) -> None:
        """Idempotently request cancellation and wait for runtime termination."""

    @abstractmethod
    async def execute_in_environment(
        self,
        session_id: str,
        command: CommandExecutionSpec,
    ) -> CommandResult:
        """Execute a verifier command in a completed Task's retained environment."""

    @abstractmethod
    async def cleanup(self, session_id: str) -> None:
        """Idempotently release one completed execution and its environment."""

    @abstractmethod
    async def close(self) -> None:
        """Cancel all executions and release all Backend-level resources."""

    async def __aenter__(self) -> ExecutionBackend:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()
