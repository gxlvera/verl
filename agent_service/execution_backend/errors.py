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

"""ExecutionBackend-specific exceptions."""


class ExecutionBackendError(Exception):
    """Base exception for Backend control-plane failures."""


class BackendNotRunningError(ExecutionBackendError):
    """Raised when an operation requires a started Backend."""


class UnknownSessionError(ExecutionBackendError, KeyError):
    """Raised when the Backend has never observed a session ID."""


class SessionLaunchConflictError(ExecutionBackendError):
    """Raised when one session ID is reused with a different execution spec."""


class SessionCleanedUpError(ExecutionBackendError):
    """Raised when an operation targets an already-cleaned session."""


class ExecutionStillRunningError(ExecutionBackendError):
    """Raised when post-execution work is requested before execution finishes."""


class WorkerProcessError(ExecutionBackendError):
    """Raised when a coroutine worker cannot serve a control request."""
