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

import pytest

from agent_service.proxy import UnsupportedRequestError
from agent_service.proxy.adapters import AnthropicMessagesAdapter, OpenAIChatCompletionsAdapter
from agent_service.proxy.canonical import Canonicalizer, common_prefix_length, exact_prefix_length


def test_openai_json_string_and_object_tool_arguments_are_equivalent():
    adapter = OpenAIChatCompletionsAdapter()
    canonicalizer = Canonicalizer()
    base = {
        "messages": [
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "verl"}'},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    }
    object_arguments = {
        **base,
        "messages": [
            base["messages"][0],
            {
                **base["messages"][1],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": {"q": "verl"}},
                    }
                ],
            },
        ],
    }

    left = canonicalizer.canonicalize(adapter.adapt_request(base))
    right = canonicalizer.canonicalize(adapter.adapt_request(object_arguments))

    assert left.messages == right.messages
    assert left.tools_fingerprint == right.tools_fingerprint


def test_equivalent_anthropic_and_openai_text_and_tools_canonicalize_identically():
    openai = OpenAIChatCompletionsAdapter().adapt_request(
        {
            "model": "policy",
            "messages": [
                {"role": "system", "content": "be useful"},
                {"role": "user", "content": "hello"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "max_tokens": 10,
        }
    )
    anthropic = AnthropicMessagesAdapter().adapt_request(
        {
            "model": "policy",
            "system": "be useful",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {"type": "object"},
                }
            ],
            "max_tokens": 10,
        }
    )

    canonicalizer = Canonicalizer()
    openai_prompt = canonicalizer.canonicalize(openai)
    anthropic_prompt = canonicalizer.canonicalize(anthropic)

    assert openai_prompt.messages == anthropic_prompt.messages
    assert openai_prompt.tools == anthropic_prompt.tools
    assert openai_prompt.tools_fingerprint == anthropic_prompt.tools_fingerprint
    assert openai_prompt.messages[0].to_dict()["content"] == "be useful"
    assert openai_prompt.messages[1].to_dict()["content"] == "hello"


def test_tool_order_and_schema_changes_are_routing_significant():
    adapter = OpenAIChatCompletionsAdapter()
    canonicalizer = Canonicalizer()
    search = {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}
    read = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
    request = {"messages": [{"role": "user", "content": "hello"}], "tools": [search, read]}
    reversed_request = {**request, "tools": [read, search]}
    changed_request = {
        **request,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "required": ["q"]},
                },
            },
            read,
        ],
    }

    base = canonicalizer.canonicalize(adapter.adapt_request(request))
    reversed_prompt = canonicalizer.canonicalize(adapter.adapt_request(reversed_request))
    changed = canonicalizer.canonicalize(adapter.adapt_request(changed_request))

    assert base.tools_fingerprint != reversed_prompt.tools_fingerprint
    assert base.tools_fingerprint != changed.tools_fingerprint


def test_prefix_helpers_confirm_structure_after_hash_indexing():
    adapter = OpenAIChatCompletionsAdapter()
    canonicalizer = Canonicalizer()
    first = canonicalizer.canonicalize(adapter.adapt_request({"messages": [{"role": "user", "content": "one"}]}))
    second = canonicalizer.canonicalize(
        adapter.adapt_request(
            {
                "messages": [
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                ]
            }
        )
    )

    prefixes = canonicalizer.prefix_hashes(second.messages)
    assert prefixes[0].rolling_hash == canonicalizer.prefix_hashes(first.messages)[0].rolling_hash
    assert exact_prefix_length(first.messages, second.messages) == 1
    assert exact_prefix_length(second.messages, first.messages) is None
    assert common_prefix_length(first.messages, second.messages) == 1


def test_configured_wire_only_field_can_be_ignored_without_mutating_source():
    source = {"messages": [{"role": "user", "content": "hello", "wire_trace_id": "one"}]}
    other = {"messages": [{"role": "user", "content": "hello", "wire_trace_id": "two"}]}
    adapter = OpenAIChatCompletionsAdapter()
    canonicalizer = Canonicalizer(ignored_message_fields=["wire_trace_id"])

    assert (
        canonicalizer.canonicalize(adapter.adapt_request(source)).messages
        == canonicalizer.canonicalize(adapter.adapt_request(other)).messages
    )
    assert source["messages"][0]["wire_trace_id"] == "one"


def test_protocol_controls_are_not_forwarded_as_backend_sampling_parameters():
    openai = OpenAIChatCompletionsAdapter().adapt_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": "auto",
            "response_format": {"type": "text"},
            "max_completion_tokens": 7,
        }
    )
    anthropic = AnthropicMessagesAdapter().adapt_request(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": False},
            "thinking": {"type": "disabled"},
            "max_tokens": 7,
        }
    )

    assert openai.sampling_params == {"max_tokens": 7}
    assert anthropic.sampling_params == {"max_tokens": 7}


def test_unsupported_protocol_controls_fail_before_upstream_generation():
    with pytest.raises(UnsupportedRequestError, match="tool_choice"):
        OpenAIChatCompletionsAdapter().adapt_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "tool_choice": "required",
            }
        )
    with pytest.raises(UnsupportedRequestError, match="thinking"):
        AnthropicMessagesAdapter().adapt_request(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
            }
        )
