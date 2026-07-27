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

"""Strict Agent Service adapter for the project-owned Continuous Token API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


class ContinuousTokenBuilderProtocol(Protocol):
    """The stable subset of the project Continuous Token builder used by Proxy."""

    def build_initial_tokens(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]: ...

    def merge_non_assistant_tokens(
        self,
        previous_messages: list[dict[str, Any]],
        updated_messages: list[dict[str, Any]],
        runtime_token_ids: list[int],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any: ...

    def merge_assistant_tokens(
        self,
        runtime_token_ids: list[int],
        assistant_token_ids: list[int],
    ) -> Any: ...


class ContinuousTokenIntegrationError(RuntimeError):
    """Raised before inference when exact token continuity cannot be preserved."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_reason = "continuous_token_merge_failed"
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class ContinuousTokenMerge:
    """Validated token input for one upstream generation."""

    context_ids: tuple[int, ...]
    appended_context_ids: tuple[int, ...]
    inserted_boundary_ids: tuple[int, ...] = ()
    removed_prefix_ids: tuple[int, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def _copy_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise TypeError("Continuous Token messages must be mappings")
        copied.append(dict(message))
    return copied


def _copy_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    copied: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            raise TypeError("Continuous Token tools must be mappings")
        copied.append(dict(tool))
    return copied


def _normalize_token_ids(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ContinuousTokenIntegrationError(
            f"{field_name} must be a sequence of token IDs",
            diagnostics={"field": field_name, "actual_type": type(value).__name__},
        )
    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ContinuousTokenIntegrationError(
                f"{field_name}[{index}] must be a non-negative integer",
                diagnostics={"field": field_name, "index": index, "value": repr(token_id)},
            )
        token_ids.append(token_id)
    return tuple(token_ids)


class ContinuousTokenCodec:
    """Fail-closed wrapper enforcing the Proxy exact-prefix invariants."""

    def __init__(self, builder: ContinuousTokenBuilderProtocol):
        self._builder = builder

    def encode_initial(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ContinuousTokenMerge:
        message_list = _copy_messages(messages)
        tool_list = _copy_tools(tools)
        try:
            context_ids = _normalize_token_ids(
                self._builder.build_initial_tokens(message_list, tools=tool_list),
                "initial token IDs",
            )
        except ContinuousTokenIntegrationError:
            raise
        except Exception as exc:
            raise ContinuousTokenIntegrationError(
                "Continuous Token initial encoding failed",
                diagnostics={"operation": "build_initial_tokens", "error": str(exc)},
            ) from exc
        return ContinuousTokenMerge(
            context_ids=context_ids,
            appended_context_ids=context_ids,
            diagnostics={"operation": "initial_encode"},
        )

    def merge_non_assistant(
        self,
        *,
        exact_prefix_ids: Sequence[int],
        previous_messages: Sequence[Mapping[str, Any]],
        updated_messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ContinuousTokenMerge:
        prefix = _normalize_token_ids(exact_prefix_ids, "exact prefix token IDs")
        previous = _copy_messages(previous_messages)
        updated = _copy_messages(updated_messages)
        tool_list = _copy_tools(tools)
        try:
            result = self._builder.merge_non_assistant_tokens(
                previous,
                updated,
                list(prefix),
                tools=tool_list,
            )
            context_ids = _normalize_token_ids(getattr(result, "token_ids", None), "merged token IDs")
            inserted_boundary_ids = _normalize_token_ids(
                getattr(result, "inserted_token_ids", ()),
                "inserted boundary token IDs",
            )
            removed_prefix_token_count = getattr(result, "removed_prefix_token_count", 0)
        except ContinuousTokenIntegrationError:
            raise
        except Exception as exc:
            raise ContinuousTokenIntegrationError(
                "Continuous Token continuation merge failed",
                diagnostics={"operation": "merge_non_assistant_tokens", "error": str(exc)},
            ) from exc

        if not isinstance(removed_prefix_token_count, int) or isinstance(removed_prefix_token_count, bool):
            raise ContinuousTokenIntegrationError(
                "Continuous Token returned an invalid removed-prefix count",
                diagnostics={"removed_prefix_token_count": repr(removed_prefix_token_count)},
            )
        if removed_prefix_token_count < 0 or removed_prefix_token_count > len(prefix):
            raise ContinuousTokenIntegrationError(
                "Continuous Token returned an out-of-range removed-prefix count",
                diagnostics={
                    "removed_prefix_token_count": removed_prefix_token_count,
                    "exact_prefix_length": len(prefix),
                },
            )
        retained_prefix_length = len(prefix) - removed_prefix_token_count
        retained_prefix = prefix[:retained_prefix_length]
        if context_ids[:retained_prefix_length] != retained_prefix:
            mismatch_index = next(
                (
                    index
                    for index, (expected, actual) in enumerate(zip(retained_prefix, context_ids, strict=False))
                    if expected != actual
                ),
                min(len(retained_prefix), len(context_ids)),
            )
            raise ContinuousTokenIntegrationError(
                "Continuous Token changed tokens outside its declared tail-boundary removal",
                diagnostics={
                    "exact_prefix_length": len(prefix),
                    "retained_prefix_length": retained_prefix_length,
                    "removed_prefix_token_count": removed_prefix_token_count,
                    "merged_length": len(context_ids),
                    "first_mismatch_index": mismatch_index,
                },
            )

        removed_prefix_ids = prefix[retained_prefix_length:]
        return ContinuousTokenMerge(
            context_ids=context_ids,
            appended_context_ids=context_ids[retained_prefix_length:],
            inserted_boundary_ids=inserted_boundary_ids,
            removed_prefix_ids=removed_prefix_ids,
            diagnostics={
                "operation": "merge_non_assistant_tokens",
                "exact_prefix_length": len(prefix),
                "retained_prefix_length": retained_prefix_length,
                "removed_prefix_token_count": removed_prefix_token_count,
                "removed_prefix_token_ids": list(removed_prefix_ids),
                "appended_context_length": len(context_ids) - retained_prefix_length,
                "inserted_boundary_length": len(inserted_boundary_ids),
            },
        )

    def append_assistant(
        self,
        *,
        context_ids: Sequence[int],
        assistant_token_ids: Sequence[int],
    ) -> tuple[int, ...]:
        context = _normalize_token_ids(context_ids, "backend context token IDs")
        assistant = _normalize_token_ids(assistant_token_ids, "assistant output token IDs")
        expected = context + assistant
        try:
            result = self._builder.merge_assistant_tokens(list(context), list(assistant))
            merged = _normalize_token_ids(getattr(result, "token_ids", None), "assistant-merged token IDs")
        except ContinuousTokenIntegrationError:
            raise
        except Exception as exc:
            raise ContinuousTokenIntegrationError(
                "Continuous Token assistant merge failed",
                diagnostics={"operation": "merge_assistant_tokens", "error": str(exc)},
            ) from exc
        if merged != expected:
            raise ContinuousTokenIntegrationError(
                "Continuous Token changed tokens while appending the assistant output",
                diagnostics={
                    "context_length": len(context),
                    "assistant_length": len(assistant),
                    "merged_length": len(merged),
                },
            )
        return merged


def create_continuous_token_codec(
    tokenizer: Any,
    *,
    model_family: str = "auto",
    model_path: str | None = None,
    tokenizer_name_or_path: str | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
    **builder_kwargs: Any,
) -> ContinuousTokenCodec:
    """Create the Proxy adapter without importing verl at module import time."""

    from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

    builder = create_continuous_token_builder(
        tokenizer,
        model_family=model_family,
        model_path=model_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        chat_template_kwargs=dict(chat_template_kwargs or {}),
        **builder_kwargs,
    )
    return ContinuousTokenCodec(builder)
