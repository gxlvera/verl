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

"""Pluggable Agent Service execution backends."""

from .base import ExecutionBackend
from .errors import (
    BackendNotRunningError,
    ExecutionBackendError,
    ExecutionStillRunningError,
    SessionCleanedUpError,
    SessionLaunchConflictError,
    UnknownSessionError,
    WorkerProcessError,
)
from .factory import create_execution_backend
from .local import (
    CoroutineRuntimeConfig,
    LocalExecutionBackend,
    LocalExecutionBackendConfig,
    ProcessRuntimeConfig,
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
)

__all__ = [
    "BackendNotRunningError",
    "CommandExecutionSpec",
    "CommandResult",
    "CoroutineLaunchSpec",
    "CoroutineRuntimeConfig",
    "ExecutionBackend",
    "ExecutionBackendError",
    "ExecutionError",
    "ExecutionResult",
    "ExecutionSnapshot",
    "ExecutionState",
    "ExecutionStillRunningError",
    "LaunchAck",
    "LocalEnvironmentSpec",
    "LocalExecutionBackend",
    "LocalExecutionBackendConfig",
    "LocalRuntimeKind",
    "ProcessLaunchSpec",
    "ProcessRuntimeConfig",
    "SessionCleanedUpError",
    "SessionLaunchConflictError",
    "TaskExecutionSpec",
    "UnknownSessionError",
    "WorkerProcessError",
    "create_execution_backend",
]
