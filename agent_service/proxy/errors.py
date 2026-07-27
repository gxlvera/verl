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

"""Stable Proxy error taxonomy shared by control and data planes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ProxyError(RuntimeError):
    code = "proxy_error"
    http_status = 500
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        state_mutated: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.request_id = request_id
        self.state_mutated = state_mutated
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
            "state_mutated": self.state_mutated,
        }
        if self.session_id is not None:
            error["session_id"] = self.session_id
        if self.request_id is not None:
            error["request_id"] = self.request_id
        if self.details:
            error["details"] = dict(self.details)
        return {"error": error}


class InvalidSessionTokenError(ProxyError):
    code = "invalid_session_token"
    http_status = 401


class UnknownSessionError(ProxyError):
    code = "unknown_session"
    http_status = 404


class MalformedRequestError(ProxyError):
    code = "malformed_request"
    http_status = 400


class UnsupportedRequestError(ProxyError):
    code = "unsupported_request"
    http_status = 422


class SessionConflictError(ProxyError):
    code = "session_conflict"
    http_status = 409


class SessionInactiveError(ProxyError):
    code = "session_inactive"
    http_status = 409


class SessionBudgetExceededError(ProxyError):
    code = "session_budget_exceeded"
    http_status = 429


class UpstreamGenerationError(ProxyError):
    code = "upstream_generation_failed"
    http_status = 502
    retryable = True


class UpstreamTimeoutError(ProxyError):
    code = "upstream_timeout"
    http_status = 504
    retryable = True


class ContinuousTokenMergeError(ProxyError):
    code = "continuous_token_merge_failed"
    http_status = 500


class TokenStateCorruptionError(ProxyError):
    code = "token_state_corruption"
    http_status = 500
