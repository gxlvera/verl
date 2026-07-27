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

"""Versioned canonical prompt normalization used before lineage routing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .models import (
    CANONICALIZATION_VERSION,
    CanonicalMessage,
    CanonicalPrompt,
    CanonicalTool,
    InternalGenerationRequest,
    MessagePrefix,
)

_JSON_ARGUMENT_KEYS = frozenset({"arguments"})


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _normalize_json_value(value: Any, *, parent_key: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_json_value(item, parent_key=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, str) and parent_key in _JSON_ARGUMENT_KEYS:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, dict | list):
            return _normalize_json_value(parsed)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise TypeError(f"Canonical values must be JSON-compatible, got {type(value).__name__}")


def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence) or isinstance(content, bytes | bytearray):
        raise TypeError("message content must be a string, list, or null")

    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(content):
        if isinstance(block, str):
            blocks.append({"type": "text", "text": block})
            continue
        if not isinstance(block, Mapping):
            raise TypeError(f"message content block {index} must be a mapping or string")
        normalized = _normalize_json_value(block)
        block_type = normalized.get("type")
        if block_type in {"input_text", "output_text"}:
            normalized["type"] = "text"
        blocks.append(normalized)
    if all(set(block) == {"type", "text"} and block["type"] == "text" for block in blocks):
        return "".join(str(block["text"]) for block in blocks)
    return blocks


def normalize_message(
    message: Mapping[str, Any],
    *,
    ignored_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise TypeError("message must be a mapping")
    role = message.get("role")
    if not isinstance(role, str) or not role:
        raise ValueError("message.role must be a non-empty string")

    normalized: dict[str, Any] = {"role": role}
    for key, value in message.items():
        if key == "role" or key in ignored_fields:
            continue
        if key == "content":
            normalized["content"] = _normalize_content(value)
        else:
            normalized[str(key)] = _normalize_json_value(value, parent_key=str(key))
    normalized.setdefault("content", [])
    if role == "assistant" and normalized.get("tool_calls") and normalized["content"] == "":
        # OpenAI/model parsers often represent a tool-only assistant as an
        # empty string, while Anthropic round-trips it as no text block.
        normalized["content"] = []
    return normalized


def normalize_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(tool, Mapping):
        raise TypeError("tool must be a mapping")
    normalized = _normalize_json_value(tool)
    if normalized.get("type") != "function":
        raise ValueError("V0 supports only function tools")
    function = normalized.get("function")
    if not isinstance(function, dict):
        raise ValueError("tool.function must be a mapping")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool.function.name must be a non-empty string")
    parameters = function.get("parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("tool.function.parameters must be a mapping when present")
    return normalized


def fingerprint_tools(
    tools: Sequence[CanonicalTool],
    *,
    canonicalization_version: str = CANONICALIZATION_VERSION,
) -> str:
    digest = hashlib.sha256()
    digest.update(f"agent-service-tools:{canonicalization_version}\0".encode())
    for tool in tools:
        encoded = tool.canonical_json.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def message_prefixes(
    messages: Sequence[CanonicalMessage],
    *,
    canonicalization_version: str = CANONICALIZATION_VERSION,
) -> tuple[MessagePrefix, ...]:
    digest = hashlib.sha256(f"agent-service-messages:{canonicalization_version}\0".encode())
    prefixes: list[MessagePrefix] = []
    for index, message in enumerate(messages, start=1):
        encoded = message.canonical_json.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        prefixes.append(MessagePrefix(message_count=index, rolling_hash=digest.hexdigest()))
    return tuple(prefixes)


def exact_prefix_length(
    prefix: Sequence[CanonicalMessage],
    value: Sequence[CanonicalMessage],
) -> int | None:
    if len(prefix) > len(value):
        return None
    for index, message in enumerate(prefix):
        if message != value[index]:
            return None
    return len(prefix)


def common_prefix_length(
    left: Sequence[CanonicalMessage],
    right: Sequence[CanonicalMessage],
) -> int:
    length = 0
    for left_message, right_message in zip(left, right, strict=False):
        if left_message != right_message:
            break
        length += 1
    return length


class Canonicalizer:
    """Create immutable routing inputs without rendering a chat template."""

    def __init__(
        self,
        *,
        version: str = CANONICALIZATION_VERSION,
        ignored_message_fields: Sequence[str] = (),
    ) -> None:
        if not isinstance(version, str) or not version:
            raise ValueError("canonicalization version must be a non-empty string")
        if any(not isinstance(field, str) or not field for field in ignored_message_fields):
            raise ValueError("ignored_message_fields must contain non-empty strings")
        self.version = version
        self.ignored_message_fields = frozenset(ignored_message_fields)

    def canonicalize(self, request: InternalGenerationRequest) -> CanonicalPrompt:
        messages = tuple(self.canonicalize_message(message) for message in request.messages)
        tools = tuple(CanonicalTool.from_dict(normalize_tool(tool)) for tool in request.tools)
        return CanonicalPrompt(
            messages=messages,
            tools=tools,
            tools_fingerprint=fingerprint_tools(tools, canonicalization_version=self.version),
            canonicalization_version=self.version,
        )

    def prefix_hashes(self, messages: Sequence[CanonicalMessage]) -> tuple[MessagePrefix, ...]:
        return message_prefixes(messages, canonicalization_version=self.version)

    def canonicalize_message(self, message: Mapping[str, Any]) -> CanonicalMessage:
        return CanonicalMessage.from_dict(normalize_message(message, ignored_fields=self.ignored_message_fields))
