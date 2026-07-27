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

import pytest

from agent_service.proxy.continuous_tokens import ContinuousTokenCodec, ContinuousTokenIntegrationError


@dataclass
class _MergeResult:
    token_ids: list[int]
    inserted_token_ids: list[int] = field(default_factory=list)
    removed_prefix_token_count: int = 0


class _Builder:
    def build_initial_tokens(self, messages, *, tools=None):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tools == [{"type": "function", "function": {"name": "search"}}]
        return [10, 11]

    def merge_non_assistant_tokens(self, previous_messages, updated_messages, runtime_token_ids, *, tools=None):
        assert updated_messages[: len(previous_messages)] == previous_messages
        return _MergeResult(runtime_token_ids + [12, 13], inserted_token_ids=[12])

    def merge_assistant_tokens(self, runtime_token_ids, assistant_token_ids):
        return _MergeResult(runtime_token_ids + assistant_token_ids)


def test_continuous_token_codec_preserves_exact_prefix():
    codec = ContinuousTokenCodec(_Builder())
    tools = [{"type": "function", "function": {"name": "search"}}]
    initial = codec.encode_initial([{"role": "user", "content": "hello"}], tools=tools)
    assert initial.context_ids == (10, 11)

    continuation = codec.merge_non_assistant(
        exact_prefix_ids=(10, 11, 90),
        previous_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
        ],
        updated_messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ],
        tools=tools,
    )

    assert continuation.context_ids == (10, 11, 90, 12, 13)
    assert continuation.appended_context_ids == (12, 13)
    assert continuation.inserted_boundary_ids == (12,)
    assert codec.append_assistant(context_ids=continuation.context_ids, assistant_token_ids=(91, 92)) == (
        10,
        11,
        90,
        12,
        13,
        91,
        92,
    )


def test_continuous_token_codec_allows_declared_tail_boundary_removal():
    class _RemovingBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            return _MergeResult(runtime_token_ids[:-1] + [12], removed_prefix_token_count=1)

    codec = ContinuousTokenCodec(_RemovingBuilder())
    merge = codec.merge_non_assistant(
        exact_prefix_ids=(10, 11, 90),
        previous_messages=[{"role": "user", "content": "hello"}],
        updated_messages=[
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "next"},
        ],
    )

    assert merge.context_ids == (10, 11, 12)
    assert merge.removed_prefix_ids == (90,)
    assert merge.appended_context_ids == (12,)
    assert merge.diagnostics["removed_prefix_token_count"] == 1


def test_continuous_token_codec_rejects_rewrite_outside_declared_tail_removal():
    class _RemovingAndRewritingBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            return _MergeResult([10, 999, 12], removed_prefix_token_count=1)

    codec = ContinuousTokenCodec(_RemovingAndRewritingBuilder())
    with pytest.raises(ContinuousTokenIntegrationError, match="outside its declared"):
        codec.merge_non_assistant(
            exact_prefix_ids=(10, 11, 90),
            previous_messages=[{"role": "user", "content": "hello"}],
            updated_messages=[
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "next"},
            ],
        )


def test_continuous_token_codec_rejects_prefix_rewrite():
    class _RewritingBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            return _MergeResult([10, 999, *runtime_token_ids[2:], 12])

    codec = ContinuousTokenCodec(_RewritingBuilder())
    with pytest.raises(ContinuousTokenIntegrationError, match="outside its declared"):
        codec.merge_non_assistant(
            exact_prefix_ids=(10, 11, 90),
            previous_messages=[{"role": "user", "content": "hello"}],
            updated_messages=[
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "next"},
            ],
        )


def test_continuous_token_codec_wraps_builder_failure_with_stable_reason():
    class _FailingBuilder(_Builder):
        def merge_non_assistant_tokens(
            self,
            previous_messages,
            updated_messages,
            runtime_token_ids,
            *,
            tools=None,
        ):
            raise ValueError("template suffix is not stable")

    codec = ContinuousTokenCodec(_FailingBuilder())
    with pytest.raises(ContinuousTokenIntegrationError) as exc_info:
        codec.merge_non_assistant(
            exact_prefix_ids=(10, 11, 90),
            previous_messages=[{"role": "user", "content": "hello"}],
            updated_messages=[
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "next"},
            ],
        )

    assert exc_info.value.failure_reason == "continuous_token_merge_failed"
    assert exc_info.value.diagnostics["operation"] == "merge_non_assistant_tokens"
