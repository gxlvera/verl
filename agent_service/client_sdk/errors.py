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

"""Errors exposed by the Agent Service client SDK."""

from __future__ import annotations


class AgentServiceError(Exception):
    """Base exception raised by the Agent Service driver SDK."""


class AgentServiceClosedError(AgentServiceError):
    """Raised when an operation is attempted on a closed client or executor."""


class AgentServiceConnectionError(AgentServiceError):
    """Raised when the Agent Service cannot be reached."""


class AgentServiceTimeoutError(AgentServiceError, TimeoutError):
    """Raised when a client-side deadline expires."""


class AgentServiceProtocolError(AgentServiceError):
    """Raised when the service returns a response that violates the wire contract."""


class MaxInFlightTasksError(AgentServiceError):
    """Raised when submitting would exceed the executor's local in-flight limit."""

    def __init__(self, max_in_flight: int):
        super().__init__(f"AgentExecutor max_in_flight limit reached: {max_in_flight}")
        self.max_in_flight = max_in_flight
