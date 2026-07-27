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

"""Internal exact-token HTTP route consumed by Agent Service Proxy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder

AGENT_SERVICE_GENERATE_PATH = "/agent_service/generate"


def _json_value(value: Any) -> Any:
    """Convert rollout tensors/arrays to JSON without changing their shape."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return value


def _token_ids(value: Any) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise HTTPException(status_code=400, detail="prompt_ids must be an array of token IDs")
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise HTTPException(
                status_code=400,
                detail=f"prompt_ids[{index}] must be a non-negative integer",
            )
        token_ids.append(token_id)
    if not token_ids:
        raise HTTPException(status_code=400, detail="prompt_ids must not be empty")
    return token_ids


def register_agent_service_generate_route(
    app: FastAPI,
    generate: Callable[..., Awaitable[Any]],
    *,
    path: str = AGENT_SERVICE_GENERATE_PATH,
) -> None:
    """Register one backend-neutral token-in/token-out route."""

    if any(getattr(route, "path", None) == path for route in app.routes):
        raise RuntimeError(f"HTTP route {path!r} is already registered")

    async def agent_service_generate(request: Request) -> dict[str, Any]:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
        if not isinstance(body, Mapping):
            raise HTTPException(status_code=400, detail="request body must be a JSON object")
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise HTTPException(status_code=400, detail="request_id must be a non-empty string")
        sampling_params = body.get("sampling_params")
        if not isinstance(sampling_params, Mapping):
            raise HTTPException(status_code=400, detail="sampling_params must be a JSON object")

        output = await generate(
            prompt_ids=_token_ids(body.get("prompt_ids")),
            sampling_params=dict(sampling_params),
            request_id=request_id,
        )
        token_ids = getattr(output, "token_ids", None)
        log_probs = getattr(output, "log_probs", None)
        routed_experts = getattr(output, "routed_experts", None)
        stop_reason = getattr(output, "stop_reason", None)
        extra_fields = getattr(output, "extra_fields", {})
        metadata = dict(extra_fields or {})
        num_preempted = getattr(output, "num_preempted", None)
        if num_preempted is not None:
            metadata["num_preempted"] = num_preempted
        return jsonable_encoder(
            {
                "token_ids": _json_value(token_ids),
                "log_probs": _json_value(log_probs),
                "routed_experts": _json_value(routed_experts),
                "finish_reason": stop_reason or "unknown",
                "metadata": _json_value(metadata),
            }
        )

    app.add_api_route(path, agent_service_generate, methods=["POST"])
