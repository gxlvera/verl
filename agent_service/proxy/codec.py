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

"""Decode raw model output into the provider-neutral assistant schema."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from .models import CanonicalMessage
from .upstream import TokenGenerationOutput


class AssistantResponseDecoder(Protocol):
    async def decode(
        self,
        output: TokenGenerationOutput,
        *,
        tools: Sequence[Mapping[str, Any]],
        request_id: str,
    ) -> CanonicalMessage: ...


class TextResponseDecoder:
    """Default decoder for text-only model output."""

    def __init__(self, tokenizer: Any, *, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens

    async def decode(
        self,
        output: TokenGenerationOutput,
        *,
        tools: Sequence[Mapping[str, Any]],
        request_id: str,
    ) -> CanonicalMessage:
        del tools, request_id
        text = await asyncio.to_thread(
            self.tokenizer.decode,
            list(output.token_ids),
            skip_special_tokens=self.skip_special_tokens,
        )
        if not isinstance(text, str):
            raise TypeError("tokenizer.decode must return a string")
        return CanonicalMessage.from_dict({"role": "assistant", "content": text})


class VerlToolResponseDecoder:
    """Adapter for verl's existing model-specific ToolParser registry."""

    def __init__(self, tokenizer: Any, *, tool_parser_name: str) -> None:
        from verl.experimental.agent_loop.tool_parser import ToolParser

        self._tool_parser = ToolParser.get_tool_parser(tool_parser_name, tokenizer)

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        return tuple(self._tool_parser.stop_token_ids)

    async def decode(
        self,
        output: TokenGenerationOutput,
        *,
        tools: Sequence[Mapping[str, Any]],
        request_id: str,
    ) -> CanonicalMessage:
        from verl.tools.schemas import OpenAIFunctionToolSchema

        tool_schemas = [OpenAIFunctionToolSchema.model_validate(tool) for tool in tools]
        content, function_calls = await self._tool_parser.extract_tool_calls(
            list(output.token_ids),
            tools=tool_schemas,
        )
        tool_calls: list[dict[str, Any]] = []
        for index, function_call in enumerate(function_calls):
            arguments: Any = function_call.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            tool_calls.append(
                {
                    "id": function_call.tool_call_id or f"call_{request_id}_{index}",
                    "type": "function",
                    "function": {
                        "name": function_call.name,
                        "arguments": arguments,
                    },
                }
            )
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return CanonicalMessage.from_dict(message)
