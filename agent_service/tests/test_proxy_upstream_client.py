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

import httpx
import pytest

from agent_service.proxy import (
    CallableTokenGenerationClient,
    HttpTokenGenerationClient,
    TokenGenerationRequest,
    UpstreamGenerationError,
    UpstreamTimeoutError,
)


def _request():
    return TokenGenerationRequest(
        request_id="request-1",
        session_id="session-1",
        prompt_token_ids=(10, 11, 777),
        sampling_params={"max_tokens": 8, "logprobs": True},
        model_name="policy",
    )


@pytest.mark.asyncio
async def test_http_upstream_sends_only_exact_token_prompt():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(
            200,
            json={
                "token_ids": [90, 91],
                "log_probs": [-0.1, -0.2],
                "stop_reason": "completed",
                "extra_fields": {"global_steps": 3},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = HttpTokenGenerationClient(client=http_client)
        output = await client.generate(replica_endpoint="http://replica-a", request=_request())

    assert captured["url"] == "http://replica-a/agent_service/generate"
    body = __import__("json").loads(captured["body"])
    assert body["prompt_ids"] == [10, 11, 777]
    assert "messages" not in body
    assert "prompt" not in body
    assert output.token_ids == (90, 91)
    assert output.logprobs == (-0.1, -0.2)
    assert output.metadata == {"global_steps": 3}


@pytest.mark.asyncio
async def test_http_upstream_does_not_retry_or_reroute_on_failure():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(503, json={"error": "unavailable"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = HttpTokenGenerationClient(client=http_client)
        with pytest.raises(UpstreamGenerationError):
            await client.generate(replica_endpoint="http://sticky-replica", request=_request())

    assert calls == ["http://sticky-replica/agent_service/generate"]


@pytest.mark.asyncio
async def test_http_upstream_requires_logprobs_for_generated_tokens():
    def handler(request):
        return httpx.Response(200, json={"token_ids": [90], "finish_reason": "stop"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = HttpTokenGenerationClient(client=http_client)
        with pytest.raises(UpstreamGenerationError, match="omitted output logprobs"):
            await client.generate(replica_endpoint="http://replica", request=_request())


@pytest.mark.asyncio
async def test_callable_upstream_maps_timeout():
    async def generate(**kwargs):
        raise TimeoutError

    client = CallableTokenGenerationClient(generate)
    with pytest.raises(UpstreamTimeoutError):
        await client.generate(replica_endpoint="ray://replica", request=_request())


@pytest.mark.asyncio
async def test_http_upstream_maps_transport_timeout():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = HttpTokenGenerationClient(client=http_client)
        with pytest.raises(UpstreamTimeoutError):
            await client.generate(replica_endpoint="http://replica", request=_request())
