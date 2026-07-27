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

"""Anthropic Messages request normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import MalformedRequestError, UnsupportedRequestError
from ..models import FrontendProtocol, InternalGenerationRequest

_RESERVED_FIELDS = frozenset({"messages", "tools", "stream", "model", "system", "metadata"})
_SUPPORTED_SAMPLING_FIELDS = frozenset(
    {
        "max_tokens",
        "stop_sequences",
        "temperature",
        "thinking",
        "tool_choice",
        "top_k",
        "top_p",
    }
)
_BACKEND_SAMPLING_FIELDS = frozenset(
    {
        "max_tokens",
        "stop_sequences",
        "temperature",
        "top_k",
        "top_p",
    }
)


def _validate_protocol_controls(body: Mapping[str, Any]) -> None:
    thinking = body.get("thinking")
    if thinking is not None and not (
        isinstance(thinking, Mapping) and thinking.get("type") == "disabled" and set(thinking) == {"type"}
    ):
        raise UnsupportedRequestError("Anthropic thinking is not supported by Agent Service V0")

    tool_choice = body.get("tool_choice")
    if tool_choice is None:
        return
    if not isinstance(tool_choice, Mapping):
        raise UnsupportedRequestError("Anthropic tool_choice must use the default auto mode")
    allowed_keys = {"type", "disable_parallel_tool_use"}
    if (
        set(tool_choice) - allowed_keys
        or tool_choice.get("type") != "auto"
        or tool_choice.get("disable_parallel_tool_use") not in (None, False)
    ):
        raise UnsupportedRequestError("Only Anthropic tool_choice type='auto' is supported")


def _anthropic_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, Sequence) or isinstance(content, bytes | bytearray):
        raise MalformedRequestError("Anthropic message content must be a string or list")
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise MalformedRequestError("Every Anthropic content block must be a JSON object")
        blocks.append(dict(block))
    return blocks


def _convert_anthropic_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role not in {"user", "assistant"}:
        raise MalformedRequestError("Anthropic message role must be user or assistant")
    blocks = _anthropic_content_blocks(message.get("content"))

    converted: list[dict[str, Any]] = []
    current_content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []

    def flush_regular_message() -> None:
        nonlocal current_content, tool_calls
        if not current_content and not tool_calls:
            return
        normalized: dict[str, Any] = {"role": role, "content": current_content}
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        converted.append(normalized)
        current_content = []
        tool_calls = []

    for block in blocks:
        block_type = block.get("type")
        if role == "assistant" and block_type == "tool_use":
            tool_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input", {})
            if not isinstance(tool_id, str) or not tool_id or not isinstance(name, str) or not name:
                raise MalformedRequestError("Anthropic tool_use requires non-empty id and name")
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        elif role == "user" and block_type == "tool_result":
            flush_regular_message()
            tool_call_id = block.get("tool_use_id")
            if not isinstance(tool_call_id, str) or not tool_call_id:
                raise MalformedRequestError("Anthropic tool_result requires tool_use_id")
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": block.get("content", ""),
                }
            )
        else:
            current_content.append(block)
    flush_regular_message()
    if not converted:
        converted.append({"role": role, "content": []})
    return converted


def _convert_system(system: Any) -> list[dict[str, Any]]:
    if system is None:
        return []
    return [{"role": "system", "content": _anthropic_content_blocks(system)}]


def _convert_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise MalformedRequestError("Anthropic tool name must be a non-empty string")
    function: dict[str, Any] = {"name": name}
    if "description" in tool:
        function["description"] = tool["description"]
    if "input_schema" in tool:
        function["parameters"] = tool["input_schema"]
    return {"type": "function", "function": function}


class AnthropicMessagesAdapter:
    protocol = FrontendProtocol.ANTHROPIC_MESSAGES

    def adapt_request(self, body: Mapping[str, Any]) -> InternalGenerationRequest:
        if not isinstance(body, Mapping):
            raise MalformedRequestError("Anthropic request body must be a JSON object")
        wire_messages = body.get("messages")
        if (
            not isinstance(wire_messages, Sequence)
            or isinstance(wire_messages, str | bytes | bytearray)
            or not wire_messages
        ):
            raise MalformedRequestError("Anthropic messages must be a non-empty list")
        if any(not isinstance(message, Mapping) for message in wire_messages):
            raise MalformedRequestError("Every Anthropic message must be a JSON object")

        messages = _convert_system(body.get("system"))
        for message in wire_messages:
            messages.extend(_convert_anthropic_message(message))

        wire_tools = body.get("tools", ())
        if wire_tools is None:
            wire_tools = ()
        if not isinstance(wire_tools, Sequence) or isinstance(wire_tools, str | bytes | bytearray):
            raise MalformedRequestError("Anthropic tools must be a list")
        if any(not isinstance(tool, Mapping) for tool in wire_tools):
            raise MalformedRequestError("Every Anthropic tool must be a JSON object")
        tools = [_convert_tool(tool) for tool in wire_tools]

        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise MalformedRequestError("Anthropic stream must be a boolean")
        unsupported = set(body) - _RESERVED_FIELDS - _SUPPORTED_SAMPLING_FIELDS
        if unsupported:
            raise UnsupportedRequestError(f"Unsupported Anthropic request fields: {sorted(unsupported)}")
        _validate_protocol_controls(body)
        sampling_params = {key: value for key, value in body.items() if key in _BACKEND_SAMPLING_FIELDS}
        if "stop_sequences" in sampling_params:
            sampling_params["stop"] = sampling_params.pop("stop_sequences")

        metadata: dict[str, Any] = {}
        if body.get("model") is not None:
            if not isinstance(body["model"], str):
                raise MalformedRequestError("Anthropic model must be a string")
            metadata["requested_model"] = body["model"]
        if body.get("metadata") is not None:
            if not isinstance(body["metadata"], Mapping):
                raise MalformedRequestError("Anthropic metadata must be a JSON object")
            metadata["provider_metadata"] = dict(body["metadata"])

        return InternalGenerationRequest(
            messages=tuple(messages),
            tools=tuple(tools),
            sampling_params=sampling_params,
            stream=stream,
            protocol=self.protocol,
            request_metadata=metadata,
        )
