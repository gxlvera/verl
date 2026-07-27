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

"""Token-in/token-out boundary between Proxy and rollout inference."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .errors import UpstreamGenerationError, UpstreamTimeoutError


def _validated_token_ids(value: Sequence[int], field_name: str) -> tuple[int, ...]:
    if isinstance(value, str | bytes | bytearray):
        raise TypeError(f"{field_name} must be a sequence of token IDs")
    result: list[int] = []
    for index, token_id in enumerate(value):
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"{field_name}[{index}] must be a non-negative integer")
        result.append(token_id)
    return tuple(result)


@dataclass(frozen=True)
class TokenGenerationRequest:
    """One exact-token request sent to a sticky rollout replica."""

    request_id: str
    session_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: Mapping[str, Any]
    model_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id must be a non-empty string")
        object.__setattr__(
            self,
            "prompt_token_ids",
            _validated_token_ids(self.prompt_token_ids, "prompt_token_ids"),
        )
        if not isinstance(self.sampling_params, Mapping):
            raise TypeError("sampling_params must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "sampling_params", dict(self.sampling_params))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TokenGenerationOutput:
    """Exact generation facts returned by rollout inference."""

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    finish_reason: str
    routed_experts: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", _validated_token_ids(self.token_ids, "token_ids"))
        if self.logprobs is not None:
            normalized_logprobs = tuple(float(value) for value in self.logprobs)
            if len(normalized_logprobs) != len(self.token_ids):
                raise ValueError("logprobs length must match token_ids length")
            if any(not math.isfinite(value) for value in normalized_logprobs):
                raise ValueError("logprobs values must be finite")
            object.__setattr__(self, "logprobs", normalized_logprobs)
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise ValueError("finish_reason must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


class UpstreamGenerator(Protocol):
    """Pluggable client for one immutable rollout-inference registry."""

    async def generate(
        self,
        *,
        replica_endpoint: str,
        request: TokenGenerationRequest,
    ) -> TokenGenerationOutput: ...


def _output_from_value(
    value: Any,
    *,
    request: TokenGenerationRequest,
    replica_endpoint: str,
) -> TokenGenerationOutput:
    if isinstance(value, TokenGenerationOutput):
        output = value
    else:
        if isinstance(value, Mapping):
            get = value.get
        else:
            get = lambda name, default=None: getattr(value, name, default)
        token_ids = get("token_ids")
        logprobs = get("logprobs")
        if logprobs is None:
            logprobs = get("log_probs")
        finish_reason = get("finish_reason")
        if finish_reason is None:
            finish_reason = get("stop_reason")
        routed_experts = get("routed_experts")
        metadata = get("metadata")
        if metadata is None:
            metadata = get("extra_fields", {})
        try:
            output = TokenGenerationOutput(
                token_ids=tuple(token_ids),
                logprobs=tuple(logprobs) if logprobs is not None else None,
                finish_reason=str(finish_reason or "unknown"),
                routed_experts=routed_experts,
                metadata=metadata or {},
            )
        except (TypeError, ValueError) as exc:
            raise UpstreamGenerationError(
                "Upstream returned an invalid token generation payload",
                session_id=request.session_id,
                request_id=request.request_id,
                details={"replica_endpoint": replica_endpoint, "error": str(exc)},
            ) from exc
    if output.token_ids and output.logprobs is None:
        raise UpstreamGenerationError(
            "Upstream omitted output logprobs",
            session_id=request.session_id,
            request_id=request.request_id,
            details={"replica_endpoint": replica_endpoint, "output_token_count": len(output.token_ids)},
        )
    return output


class HttpTokenGenerationClient:
    """HTTP client for the rollout server's exact-token generation API."""

    def __init__(
        self,
        *,
        generate_path: str = "/agent_service/generate",
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not isinstance(generate_path, str) or not generate_path.startswith("/"):
            raise ValueError("generate_path must start with '/'")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.generate_path = generate_path
        self.timeout_seconds = timeout_seconds
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def generate(
        self,
        *,
        replica_endpoint: str,
        request: TokenGenerationRequest,
    ) -> TokenGenerationOutput:
        url = f"{replica_endpoint.rstrip('/')}{self.generate_path}"
        payload = {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "prompt_ids": list(request.prompt_token_ids),
            "sampling_params": dict(request.sampling_params),
        }
        if request.model_name is not None:
            payload["model"] = request.model_name
        if request.metadata:
            payload["metadata"] = dict(request.metadata)
        try:
            response = await self._client.post(
                url,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            value = response.json()
        except httpx.TimeoutException as exc:
            raise UpstreamTimeoutError(
                "Upstream token generation timed out",
                session_id=request.session_id,
                request_id=request.request_id,
                details={"replica_endpoint": replica_endpoint},
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            raise UpstreamGenerationError(
                "Upstream token generation request failed",
                session_id=request.session_id,
                request_id=request.request_id,
                details={
                    "replica_endpoint": replica_endpoint,
                    "status_code": status_code,
                    "error": str(exc),
                },
            ) from exc
        return _output_from_value(
            value,
            request=request,
            replica_endpoint=replica_endpoint,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class CallableTokenGenerationClient:
    """Adapter for an injected in-process or Ray token-generation callable."""

    def __init__(
        self,
        generate: Callable[..., Awaitable[Any]],
        *,
        timeout_seconds: float = 120.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._generate = generate
        self.timeout_seconds = timeout_seconds

    async def generate(
        self,
        *,
        replica_endpoint: str,
        request: TokenGenerationRequest,
    ) -> TokenGenerationOutput:
        try:
            value = await asyncio.wait_for(
                self._generate(
                    replica_endpoint=replica_endpoint,
                    request_id=request.request_id,
                    session_id=request.session_id,
                    prompt_ids=list(request.prompt_token_ids),
                    sampling_params=dict(request.sampling_params),
                    model_name=request.model_name,
                    metadata=dict(request.metadata),
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as exc:
            raise UpstreamTimeoutError(
                "Upstream token generation timed out",
                session_id=request.session_id,
                request_id=request.request_id,
                details={"replica_endpoint": replica_endpoint},
            ) from exc
        except UpstreamGenerationError:
            raise
        except Exception as exc:
            raise UpstreamGenerationError(
                "Upstream token generation callable failed",
                session_id=request.session_id,
                request_id=request.request_id,
                details={"replica_endpoint": replica_endpoint, "error": str(exc)},
            ) from exc
        return _output_from_value(
            value,
            request=request,
            replica_endpoint=replica_endpoint,
        )
