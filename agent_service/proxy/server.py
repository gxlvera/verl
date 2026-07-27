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

"""FastAPI data plane for OpenAI and Anthropic-compatible Agents."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .adapters import AnthropicMessagesAdapter, OpenAIChatCompletionsAdapter
from .continuous_tokens import ContinuousTokenIntegrationError
from .errors import ContinuousTokenMergeError, MalformedRequestError, ProxyError
from .models import CanonicalMessage, DeliveryStatus
from .processor import ProcessedGeneration, ProxyRequestProcessor
from .session import ProxySession, ProxySessionManager


class ProxyHttpServer:
    """Small uvicorn lifecycle wrapper with discoverable ephemeral ports."""

    def __init__(
        self,
        app: FastAPI,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        advertised_host: str | None = None,
        ready_timeout_seconds: float = 30.0,
        log_level: str = "warning",
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if ready_timeout_seconds <= 0:
            raise ValueError("ready_timeout_seconds must be greater than zero")
        self.app = app
        self.host = host
        self.requested_port = port
        self.advertised_host = advertised_host or host
        self.ready_timeout_seconds = ready_timeout_seconds
        self.log_level = log_level
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        self._port: int | None = None

    @property
    def endpoint(self) -> str:
        if self._port is None:
            raise RuntimeError("Proxy HTTP server is not started")
        return f"http://{self.advertised_host}:{self._port}"

    async def start(self) -> str:
        if self._task is not None:
            raise RuntimeError("Proxy HTTP server is already started")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.requested_port))
        sock.listen(2048)
        sock.setblocking(False)
        self._socket = sock
        self._port = int(sock.getsockname()[1])
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self._port,
            log_level=self.log_level,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve(sockets=[sock]))

        async def wait_ready() -> None:
            assert self._server is not None and self._task is not None
            while not self._server.started:
                if self._task.done():
                    await self._task
                    raise RuntimeError("Proxy HTTP server stopped before becoming ready")
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(wait_ready(), timeout=self.ready_timeout_seconds)
        except BaseException:
            await self.close()
            raise
        return self.endpoint

    async def close(self) -> None:
        server = self._server
        task = self._task
        if server is not None:
            server.should_exit = True
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=self.ready_timeout_seconds)
            except TimeoutError:
                if server is not None:
                    server.force_exit = True
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._server = None
        self._task = None
        self._port = None


def _request_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            return token
    return request.headers.get("x-api-key")


async def _json_body(request: Request) -> Mapping[str, Any]:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedRequestError("Request body must be valid JSON") from exc
    if not isinstance(body, Mapping):
        raise MalformedRequestError("Request body must be a JSON object")
    return body


def _text_content(message: CanonicalMessage) -> str:
    content = message.to_dict().get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def _openai_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _openai_message(message: CanonicalMessage) -> dict[str, Any]:
    value = message.to_dict()
    result: dict[str, Any] = {
        "role": "assistant",
        "content": _text_content(message),
    }
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        result["tool_calls"] = []
        for tool_call in tool_calls:
            copied = dict(tool_call)
            function = dict(copied.get("function") or {})
            function["arguments"] = _openai_arguments(function.get("arguments", {}))
            copied["function"] = function
            result["tool_calls"].append(copied)
    return result


def _anthropic_content(message: CanonicalMessage) -> list[dict[str, Any]]:
    value = message.to_dict()
    content: list[dict[str, Any]] = []
    text = _text_content(message)
    if text:
        content.append({"type": "text", "text": text})
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"raw": arguments}
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.get("id"),
                    "name": function.get("name"),
                    "input": arguments,
                }
            )
    return content or [{"type": "text", "text": ""}]


def _openai_finish_reason(result: ProcessedGeneration) -> str:
    if result.response_message.to_dict().get("tool_calls"):
        return "tool_calls"
    if result.output.finish_reason in {"length", "max_tokens"}:
        return "length"
    return "stop"


def _anthropic_stop_reason(result: ProcessedGeneration) -> str:
    if result.response_message.to_dict().get("tool_calls"):
        return "tool_use"
    if result.output.finish_reason in {"length", "max_tokens"}:
        return "max_tokens"
    return "end_turn"


def _openai_response(result: ProcessedGeneration, model_name: str | None) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{result.prepared.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name or "agent-service-proxy",
        "choices": [
            {
                "index": 0,
                "message": _openai_message(result.response_message),
                "finish_reason": _openai_finish_reason(result),
            }
        ],
        "usage": {
            "prompt_tokens": len(result.prepared.context_ids),
            "completion_tokens": len(result.output.token_ids),
            "total_tokens": len(result.prepared.context_ids) + len(result.output.token_ids),
        },
    }


def _anthropic_response(result: ProcessedGeneration, model_name: str | None) -> dict[str, Any]:
    return {
        "id": f"msg_{result.prepared.request_id}",
        "type": "message",
        "role": "assistant",
        "model": model_name or "agent-service-proxy",
        "content": _anthropic_content(result.response_message),
        "stop_reason": _anthropic_stop_reason(result),
        "stop_sequence": None,
        "usage": {
            "input_tokens": len(result.prepared.context_ids),
            "output_tokens": len(result.output.token_ids),
        },
    }


def _sse(event: Mapping[str, Any], *, event_name: str | None = None) -> bytes:
    prefix = f"event: {event_name}\n" if event_name else ""
    return f"{prefix}data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()


async def _mark_stream_delivery(
    iterator: AsyncIterator[bytes],
    *,
    session: ProxySession,
    turn_id: str,
) -> AsyncIterator[bytes]:
    delivered = False
    try:
        async for chunk in iterator:
            yield chunk
        delivered = True
    except asyncio.CancelledError:
        raise
    finally:
        await session.mark_delivery(
            turn_id=turn_id,
            status=DeliveryStatus.DELIVERED if delivered else DeliveryStatus.FAILED,
        )


async def _openai_stream(result: ProcessedGeneration, model_name: str | None) -> AsyncIterator[bytes]:
    response_id = f"chatcmpl-{result.prepared.request_id}"
    base = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name or "agent-service-proxy",
    }
    yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    message = _openai_message(result.response_message)
    delta = {key: value for key, value in message.items() if key != "role"}
    if delta:
        yield _sse({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
    yield _sse(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": _openai_finish_reason(result)}],
        }
    )
    yield b"data: [DONE]\n\n"


async def _anthropic_stream(result: ProcessedGeneration, model_name: str | None) -> AsyncIterator[bytes]:
    message = _anthropic_response(result, model_name)
    content = message.pop("content")
    yield _sse(
        {
            "type": "message_start",
            "message": {**message, "content": [], "stop_reason": None, "stop_sequence": None},
        },
        event_name="message_start",
    )
    for index, block in enumerate(content):
        empty_block = {"type": block["type"]}
        if block["type"] == "text":
            empty_block["text"] = ""
            delta = {"type": "text_delta", "text": block.get("text", "")}
        else:
            empty_block.update({"id": block.get("id"), "name": block.get("name"), "input": {}})
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False, separators=(",", ":")),
            }
        yield _sse(
            {"type": "content_block_start", "index": index, "content_block": empty_block},
            event_name="content_block_start",
        )
        yield _sse(
            {"type": "content_block_delta", "index": index, "delta": delta},
            event_name="content_block_delta",
        )
        yield _sse(
            {"type": "content_block_stop", "index": index},
            event_name="content_block_stop",
        )
    yield _sse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": _anthropic_stop_reason(result), "stop_sequence": None},
            "usage": {"output_tokens": len(result.output.token_ids)},
        },
        event_name="message_delta",
    )
    yield _sse({"type": "message_stop"}, event_name="message_stop")


def create_proxy_app(
    *,
    session_manager: ProxySessionManager,
    processor: ProxyRequestProcessor,
) -> FastAPI:
    app = FastAPI()
    openai_adapter = OpenAIChatCompletionsAdapter()
    anthropic_adapter = AnthropicMessagesAdapter()

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError) -> JSONResponse:
        del request
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    async def resolve(request: Request, session_id: str) -> ProxySession:
        return await session_manager.authorize(session_id, _request_token(request))

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def openai_chat(request: Request, session_id: str):
        session = await resolve(request, session_id)
        internal = openai_adapter.adapt_request(await _json_body(request))
        request_id = request.headers.get("x-request-id") or uuid4().hex
        result = await processor.generate(session=session, request_id=request_id, request=internal)
        if internal.stream:
            stream = _mark_stream_delivery(
                _openai_stream(result, session.model_name),
                session=session,
                turn_id=result.turn.turn_id,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return JSONResponse(
            _openai_response(result, session.model_name),
            background=BackgroundTask(
                session.mark_delivery,
                turn_id=result.turn.turn_id,
                status=DeliveryStatus.DELIVERED,
            ),
        )

    @app.post("/sessions/{session_id}/v1/messages")
    async def anthropic_messages(request: Request, session_id: str):
        session = await resolve(request, session_id)
        internal = anthropic_adapter.adapt_request(await _json_body(request))
        request_id = request.headers.get("x-request-id") or uuid4().hex
        result = await processor.generate(session=session, request_id=request_id, request=internal)
        if internal.stream:
            stream = _mark_stream_delivery(
                _anthropic_stream(result, session.model_name),
                session=session,
                turn_id=result.turn.turn_id,
            )
            return StreamingResponse(stream, media_type="text/event-stream")
        return JSONResponse(
            _anthropic_response(result, session.model_name),
            background=BackgroundTask(
                session.mark_delivery,
                turn_id=result.turn.turn_id,
                status=DeliveryStatus.DELIVERED,
            ),
        )

    @app.post("/sessions/{session_id}/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request, session_id: str):
        session = await resolve(request, session_id)
        internal = anthropic_adapter.adapt_request(await _json_body(request))
        request_id = request.headers.get("x-request-id") or uuid4().hex
        try:
            prompt = session.canonicalizer.canonicalize(internal)
        except (TypeError, ValueError) as exc:
            raise MalformedRequestError(
                "Request messages or tools are malformed",
                session_id=session.session_id,
                request_id=request_id,
                details={"error": str(exc)},
            ) from exc
        try:
            encoded = session.continuous_token_codec.encode_initial(
                [message.to_dict() for message in prompt.messages],
                tools=[tool.to_dict() for tool in prompt.tools],
            )
        except ContinuousTokenIntegrationError as exc:
            raise ContinuousTokenMergeError(
                str(exc),
                session_id=session.session_id,
                request_id=request_id,
                state_mutated=False,
                details=exc.diagnostics,
            ) from exc
        return {"input_tokens": len(encoded.context_ids)}

    return app
