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

from agent_service.proxy import (
    CanonicalMessage,
    ContinuousTokenCodec,
    GenerationSpec,
    HostedProxy,
    ProxyHttpServer,
    ProxyRequestProcessor,
    ProxySessionManager,
    TokenGenerationOutput,
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
    async def generate(self, *, replica_endpoint, request):
        return TokenGenerationOutput(token_ids=(90,), logprobs=(-0.1,), finish_reason="stop")


class _Decoder:
    async def decode(self, output, *, tools, request_id):
        return CanonicalMessage.from_dict({"role": "assistant", "content": "live"})


@pytest.mark.asyncio
async def test_hosted_proxy_real_http_smoke():
    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://pending",
        continuous_token_codec=ContinuousTokenCodec(_Builder()),
        token_factory=lambda: "secret",
    )
    app = create_proxy_app(
        session_manager=manager,
        processor=ProxyRequestProcessor(upstream=_Upstream(), response_decoder=_Decoder()),
    )
    proxy = HostedProxy(
        session_manager=manager,
        http_server=ProxyHttpServer(app, port=0),
    )
    await proxy.start()
    try:
        handle = await proxy.create_session(
            session_id="session-1",
            generation_spec=GenerationSpec(),
        )
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{handle.frontend_endpoint}/v1/chat/completions",
                headers={"authorization": f"Bearer {handle.auth_token}"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "live"
        bundle = await proxy.finalize_session("session-1")
        assert len(bundle.trajectories) == 1
    finally:
        await proxy.close()


@pytest.mark.asyncio
async def test_hosted_proxy_anthropic_tool_round_trip_keeps_exact_generated_prefix():
    class _RecordingUpstream:
        def __init__(self):
            self.prompt_ids = []

        async def generate(self, *, replica_endpoint, request):
            del replica_endpoint
            self.prompt_ids.append(request.prompt_token_ids)
            token_id = 90 + len(self.prompt_ids) - 1
            return TokenGenerationOutput(token_ids=(token_id,), logprobs=(-0.1,), finish_reason="stop")

    class _ToolThenTextDecoder:
        async def decode(self, output, *, tools, request_id):
            del tools, request_id
            if output.token_ids == (90,):
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
            return CanonicalMessage.from_dict({"role": "assistant", "content": "done"})

    upstream = _RecordingUpstream()
    manager = ProxySessionManager(
        replica_endpoints=["http://replica"],
        frontend_base_url="http://pending",
        continuous_token_codec=ContinuousTokenCodec(_Builder()),
        token_factory=lambda: "secret",
    )
    app = create_proxy_app(
        session_manager=manager,
        processor=ProxyRequestProcessor(upstream=upstream, response_decoder=_ToolThenTextDecoder()),
    )
    proxy = HostedProxy(
        session_manager=manager,
        http_server=ProxyHttpServer(app, port=0),
    )
    await proxy.start()
    try:
        handle = await proxy.create_session(session_id="session-tool", generation_spec=GenerationSpec())
        tools = [
            {
                "name": "search",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ]
        async with httpx.AsyncClient() as client:
            first = await client.post(
                f"{handle.frontend_endpoint}/v1/messages",
                headers={"x-api-key": handle.auth_token},
                json={
                    "messages": [{"role": "user", "content": "search"}],
                    "tools": tools,
                    "max_tokens": 4,
                },
            )
            assert first.status_code == 200
            second = await client.post(
                f"{handle.frontend_endpoint}/v1/messages",
                headers={"x-api-key": handle.auth_token},
                json={
                    "messages": [
                        {"role": "user", "content": "search"},
                        {"role": "assistant", "content": first.json()["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "call-1",
                                    "content": "result",
                                }
                            ],
                        },
                    ],
                    "tools": tools,
                    "max_tokens": 4,
                },
            )

        assert second.status_code == 200
        assert second.json()["content"] == [{"type": "text", "text": "done"}]
        assert upstream.prompt_ids == [(10, 11), (10, 11, 90, 12)]
        bundle = await proxy.finalize_session("session-tool")
        assert len(bundle.trajectories) == 1
        assert bundle.trajectories[0].response_ids == (90, 12, 91)
        assert bundle.trajectories[0].generation_mask == (1, 0, 1)
    finally:
        await proxy.close()
