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

import asyncio
from dataclasses import dataclass, field

import httpx
import pytest

from agent_service.proxy import (
    CanonicalMessage,
    ContinuousTokenCodec,
    GenerationSpec,
    ProxyRequestProcessor,
    ProxySessionManager,
    TokenGenerationOutput,
    UpstreamGenerationError,
    create_proxy_app,
)


@dataclass
class _MergeResult:
    token_ids: list[int]
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class _Builder:
    def build_initial_tokens(self, messages, *, tools=None):
        return [10, 11]

    def merge_non_assistant_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools=None):
        return _MergeResult(runtime_token_ids + [12])

    def merge_assistant_tokens(self, runtime_token_ids, assistant_token_ids):
        return _MergeResult(runtime_token_ids + assistant_token_ids)


class _Upstream:
    def __init__(self):
        self.calls = []

    async def generate(self, *, replica_endpoint, request):
        self.calls.append((replica_endpoint, request))
        return TokenGenerationOutput(
            token_ids=(90, 91),
            logprobs=(-0.1, -0.2),
            finish_reason="stop",
        )


class _TextDecoder:
    async def decode(self, output, *, tools, request_id):
        return CanonicalMessage.from_dict({"role": "assistant", "content": "hello"})


class _ToolDecoder:
    async def decode(self, output, *, tools, request_id):
        return CanonicalMessage.from_dict(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": {"q": "verl"}},
                    }
                ],
            }
        )


async def _client(decoder=None, upstream=None):
    upstream = upstream or _Upstream()
    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://proxy",
        continuous_token_codec=ContinuousTokenCodec(_Builder()),
        token_factory=lambda: "secret",
        model_name="policy",
    )
    await manager.create_session(session_id="session-1", generation_spec=GenerationSpec())
    app = create_proxy_app(
        session_manager=manager,
        processor=ProxyRequestProcessor(
            upstream=upstream,
            response_decoder=decoder or _TextDecoder(),
        ),
    )
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    return client, manager, upstream


@pytest.mark.asyncio
async def test_openai_non_streaming_uses_exact_token_pipeline():
    client, manager, upstream = await _client()
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/chat/completions",
            headers={"authorization": "Bearer secret", "x-request-id": "request-1"},
            json={
                "model": "alias",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert upstream.calls[0][0] == "http://replica"
    assert upstream.calls[0][1].prompt_token_ids == (10, 11)
    session = await manager.get_session("session-1")
    turn = next(iter(session.chains.values())).turns[0]
    assert turn.backend_prompt_ids == (10, 11)
    assert turn.output_token_ids == (90, 91)
    assert turn.delivery_status.value == "delivered"


@pytest.mark.asyncio
async def test_anthropic_tool_use_response():
    client, _, _ = await _client(decoder=_ToolDecoder())
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/messages",
            headers={"x-api-key": "secret"},
            json={
                "model": "alias",
                "messages": [{"role": "user", "content": "search"}],
                "tools": [
                    {
                        "name": "search",
                        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
                    }
                ],
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["stop_reason"] == "tool_use"
    assert body["content"] == [
        {
            "type": "tool_use",
            "id": "call-1",
            "name": "search",
            "input": {"q": "verl"},
        }
    ]


@pytest.mark.asyncio
async def test_openai_streaming_event_order_and_delivery():
    client, manager, _ = await _client()
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/chat/completions",
            headers={"authorization": "Bearer secret"},
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.text.startswith("data: ")
    assert response.text.rstrip().endswith("data: [DONE]")
    session = await manager.get_session("session-1")
    assert next(iter(session.chains.values())).turns[0].delivery_status.value == "delivered"


@pytest.mark.asyncio
async def test_missing_auth_is_a_distinct_401_error():
    client, _, upstream = await _client()
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_session_token"
    assert upstream.calls == []


@pytest.mark.asyncio
async def test_invalid_message_shape_is_a_malformed_request_not_an_internal_error():
    client, _, upstream = await _client()
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/chat/completions",
            headers={"authorization": "Bearer secret", "x-request-id": "request-bad"},
            json={"messages": [{"content": "missing role"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_request"
    assert response.json()["error"]["request_id"] == "request-bad"
    assert upstream.calls == []


@pytest.mark.asyncio
async def test_count_tokens_reuses_adapter_canonicalizer_and_codec():
    client, _, upstream = await _client()
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/messages/count_tokens",
            headers={"x-api-key": "secret"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"input_tokens": 2}
    assert upstream.calls == []


@pytest.mark.asyncio
async def test_count_tokens_rejects_an_inactive_session():
    client, manager, upstream = await _client()
    session = await manager.get_session("session-1")
    await session.abort()
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/messages/count_tokens",
            headers={"x-api-key": "secret"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "session_inactive"
    assert upstream.calls == []


@pytest.mark.asyncio
async def test_upstream_failure_releases_reservation():
    class _FailingUpstream:
        async def generate(self, *, replica_endpoint, request):
            raise UpstreamGenerationError(
                "down",
                session_id=request.session_id,
                request_id=request.request_id,
            )

    client, manager, _ = await _client(upstream=_FailingUpstream())
    async with client:
        response = await client.post(
            "/sessions/session-1/v1/chat/completions",
            headers={"authorization": "Bearer secret"},
            json={"messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 502
    session = await manager.get_session("session-1")
    assert session.in_flight_request_ids == ()


@pytest.mark.asyncio
async def test_cancelled_upstream_request_releases_reservation():
    class _BlockingUpstream:
        def __init__(self):
            self.started = asyncio.Event()

        async def generate(self, *, replica_endpoint, request):
            del replica_endpoint, request
            self.started.set()
            await asyncio.Event().wait()

    upstream = _BlockingUpstream()
    client, manager, _ = await _client(upstream=upstream)
    async with client:
        request_task = asyncio.create_task(
            client.post(
                "/sessions/session-1/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        )
        await upstream.started.wait()
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    session = await manager.get_session("session-1")
    assert session.in_flight_request_ids == ()


@pytest.mark.asyncio
async def test_cancellation_after_generation_still_commits_token_truth():
    class _BlockingDecoder:
        def __init__(self):
            self.started = asyncio.Event()

        async def decode(self, output, *, tools, request_id):
            del output, tools, request_id
            self.started.set()
            await asyncio.Event().wait()

    decoder = _BlockingDecoder()
    client, manager, _ = await _client(decoder=decoder)
    async with client:
        request_task = asyncio.create_task(
            client.post(
                "/sessions/session-1/v1/chat/completions",
                headers={"authorization": "Bearer secret"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        )
        await decoder.started.wait()
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task

    session = await manager.get_session("session-1")
    assert session.in_flight_request_ids == ()
    turn = next(iter(session.chains.values())).turns[0]
    assert turn.output_token_ids == (90, 91)
    assert turn.metadata["response_decode_cancelled"] is True
