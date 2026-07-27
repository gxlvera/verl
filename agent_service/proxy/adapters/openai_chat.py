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

"""OpenAI Chat Completions request normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import MalformedRequestError, UnsupportedRequestError
from ..models import FrontendProtocol, InternalGenerationRequest

_RESERVED_FIELDS = frozenset({"messages", "tools", "stream", "model", "metadata"})
_SUPPORTED_SAMPLING_FIELDS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "temperature",
        "tool_choice",
        "top_logprobs",
        "top_p",
    }
)
_BACKEND_SAMPLING_FIELDS = frozenset(
    {
        "frequency_penalty",
        "max_completion_tokens",
        "max_tokens",
        "presence_penalty",
        "seed",
        "stop",
        "temperature",
        "top_p",
    }
)


def _validate_protocol_controls(body: Mapping[str, Any]) -> None:
    if body.get("logprobs") not in (None, False):
        raise UnsupportedRequestError("OpenAI response logprobs are not exposed by Agent Service V0")
    if body.get("top_logprobs") is not None:
        raise UnsupportedRequestError("OpenAI top_logprobs is not supported by Agent Service V0")
    if body.get("logit_bias") not in (None, {}):
        raise UnsupportedRequestError("OpenAI logit_bias is not supported by Agent Service V0")
    if body.get("response_format") not in (None, {"type": "text"}):
        raise UnsupportedRequestError("Only OpenAI response_format={'type': 'text'} is supported")
    if body.get("tool_choice") not in (None, "auto"):
        raise UnsupportedRequestError("Only OpenAI tool_choice='auto' is supported")


class OpenAIChatCompletionsAdapter:
    protocol = FrontendProtocol.OPENAI_CHAT_COMPLETIONS

    def adapt_request(self, body: Mapping[str, Any]) -> InternalGenerationRequest:
        if not isinstance(body, Mapping):
            raise MalformedRequestError("OpenAI request body must be a JSON object")
        messages = body.get("messages")
        if not isinstance(messages, Sequence) or isinstance(messages, str | bytes | bytearray) or not messages:
            raise MalformedRequestError("OpenAI messages must be a non-empty list")
        if any(not isinstance(message, Mapping) for message in messages):
            raise MalformedRequestError("Every OpenAI message must be a JSON object")

        tools = body.get("tools", ())
        if tools is None:
            tools = ()
        if not isinstance(tools, Sequence) or isinstance(tools, str | bytes | bytearray):
            raise MalformedRequestError("OpenAI tools must be a list")
        if any(not isinstance(tool, Mapping) for tool in tools):
            raise MalformedRequestError("Every OpenAI tool must be a JSON object")

        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise MalformedRequestError("OpenAI stream must be a boolean")
        n = body.get("n", 1)
        if not isinstance(n, int) or isinstance(n, bool) or n != 1:
            raise UnsupportedRequestError("Agent Service V0 supports only OpenAI n=1")

        unsupported = set(body) - _RESERVED_FIELDS - _SUPPORTED_SAMPLING_FIELDS
        if unsupported:
            raise UnsupportedRequestError(f"Unsupported OpenAI request fields: {sorted(unsupported)}")
        _validate_protocol_controls(body)
        sampling_params = {key: value for key, value in body.items() if key in _BACKEND_SAMPLING_FIELDS}
        if "max_completion_tokens" in sampling_params:
            if "max_tokens" in sampling_params:
                raise MalformedRequestError("Specify only one of max_tokens and max_completion_tokens")
            sampling_params["max_tokens"] = sampling_params.pop("max_completion_tokens")

        metadata: dict[str, Any] = {}
        if body.get("model") is not None:
            if not isinstance(body["model"], str):
                raise MalformedRequestError("OpenAI model must be a string")
            metadata["requested_model"] = body["model"]
        if body.get("metadata") is not None:
            if not isinstance(body["metadata"], Mapping):
                raise MalformedRequestError("OpenAI metadata must be a JSON object")
            metadata["provider_metadata"] = dict(body["metadata"])

        return InternalGenerationRequest(
            messages=tuple(dict(message) for message in messages),
            tools=tuple(dict(tool) for tool in tools),
            sampling_params=sampling_params,
            stream=stream,
            protocol=self.protocol,
            request_metadata=metadata,
        )
