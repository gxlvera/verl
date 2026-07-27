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

from dataclasses import dataclass, field

import httpx
import pytest
from fastapi import FastAPI

from verl.workers.rollout.token_http import (
    AGENT_SERVICE_GENERATE_PATH,
    register_agent_service_generate_route,
)


@dataclass
class _Output:
    token_ids: list[int]
    log_probs: list[float]
    routed_experts: object | None = None
    stop_reason: str = "stop"
    num_preempted: int | None = None
    extra_fields: dict = field(default_factory=lambda: {"global_steps": 3})


@pytest.mark.asyncio
async def test_agent_service_generate_route_is_exact_token_in_token_out():
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return _Output(token_ids=[90, 91], log_probs=[-0.1, -0.2])

    app = FastAPI()
    register_agent_service_generate_route(app, generate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://rollout",
    ) as client:
        response = await client.post(
            AGENT_SERVICE_GENERATE_PATH,
            json={
                "request_id": "request-1",
                "session_id": "session-1",
                "prompt_ids": [10, 11, 777],
                "sampling_params": {"max_tokens": 8, "logprobs": True},
            },
        )

    assert response.status_code == 200
    assert calls == [
        {
            "request_id": "request-1",
            "prompt_ids": [10, 11, 777],
            "sampling_params": {"max_tokens": 8, "logprobs": True},
        }
    ]
    assert response.json() == {
        "token_ids": [90, 91],
        "log_probs": [-0.1, -0.2],
        "routed_experts": None,
        "finish_reason": "stop",
        "metadata": {"global_steps": 3},
    }


@pytest.mark.asyncio
async def test_agent_service_generate_route_rejects_invalid_token_ids_before_generation():
    called = False

    async def generate(**kwargs):
        nonlocal called
        called = True

    app = FastAPI()
    register_agent_service_generate_route(app, generate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://rollout",
    ) as client:
        response = await client.post(
            AGENT_SERVICE_GENERATE_PATH,
            json={
                "request_id": "request-1",
                "prompt_ids": [10, -1],
                "sampling_params": {},
            },
        )

    assert response.status_code == 400
    assert called is False


@pytest.mark.asyncio
async def test_agent_service_generate_route_serializes_tensor_like_metadata():
    class _TensorLike:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    async def generate(**kwargs):
        del kwargs
        return _Output(
            token_ids=[90],
            log_probs=[-0.1],
            routed_experts=_TensorLike([[[1, 2]]]),
            num_preempted=2,
            extra_fields={"prompt_logprobs": _TensorLike([-0.3, -0.2])},
        )

    app = FastAPI()
    register_agent_service_generate_route(app, generate)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://rollout",
    ) as client:
        response = await client.post(
            AGENT_SERVICE_GENERATE_PATH,
            json={
                "request_id": "request-1",
                "prompt_ids": [10],
                "sampling_params": {},
            },
        )

    assert response.status_code == 200
    assert response.json()["routed_experts"] == [[[1, 2]]]
    assert response.json()["metadata"]["prompt_logprobs"] == [-0.3, -0.2]
    assert response.json()["metadata"]["num_preempted"] == 2
