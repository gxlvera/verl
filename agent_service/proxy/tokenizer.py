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

"""Load the tokenizer and processor artifacts owned by the Proxy."""

from __future__ import annotations

import importlib
import types
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _import_external_libs(external_libs: str | Sequence[str] | None) -> None:
    if external_libs is None:
        return
    if isinstance(external_libs, str):
        external_libs = [external_libs]
    if not isinstance(external_libs, Sequence) or any(not isinstance(name, str) or not name for name in external_libs):
        raise TypeError("model_config.external_lib must be a module name, a sequence of module names, or null")
    for module_name in external_libs:
        importlib.import_module(module_name)


def _resolve_tokenizer_source(
    model_config: Mapping[str, Any],
    source_resolver: Callable[[str], str] | None = None,
) -> tuple[str, bool]:
    if not isinstance(model_config, Mapping):
        raise TypeError("model_config must be a mapping")

    tokenizer_source = model_config.get("tokenizer_path") or model_config.get("path")
    if not isinstance(tokenizer_source, str) or not tokenizer_source:
        raise ValueError("model_config.tokenizer_path or model_config.path must be a non-empty string")

    _import_external_libs(model_config.get("external_lib"))
    if source_resolver is not None:
        tokenizer_source = source_resolver(tokenizer_source)
        if not isinstance(tokenizer_source, str) or not tokenizer_source:
            raise ValueError("source_resolver must return a non-empty tokenizer path")

    trust_remote_code = model_config.get("trust_remote_code", False)
    if not isinstance(trust_remote_code, bool):
        raise TypeError("model_config.trust_remote_code must be a boolean")
    return tokenizer_source, trust_remote_code


def _load_tokenizer(tokenizer_source: str, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if "gemma-2-2b-it" in tokenizer_source:
        tokenizer_kwargs.update({"eos_token": "<end_of_turn>", "eos_token_id": 107})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def _load_processor(tokenizer_source: str, trust_remote_code: bool) -> Any | None:
    from transformers import AutoConfig, AutoProcessor, PreTrainedTokenizerBase

    try:
        processor = AutoProcessor.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
        if isinstance(processor, PreTrainedTokenizerBase):
            return None

        config = AutoConfig.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
        processor.config = config
        model_class = None
        match processor.__class__.__name__:
            case "Qwen2VLProcessor":
                from transformers.models.qwen2_vl import Qwen2VLModel

                model_class = Qwen2VLModel
            case "Qwen2_5_VLProcessor":
                from transformers.models.qwen2_5_vl import Qwen2_5_VLModel

                model_class = Qwen2_5_VLModel
            case "Qwen3VLProcessor":
                from transformers.models.qwen3_vl import Qwen3VLModel

                model_class = Qwen3VLModel
            case "Glm4vProcessor":
                from transformers.models.glm4v import Glm4vModel

                model_class = Glm4vModel
            case "MllamaProcessor":
                pass
            case "Gemma4Processor":
                processor.validate_inputs = lambda *args, **kwargs: None
            case _:
                raise ValueError(f"Unsupported processor type: {processor.__class__.__name__}")

        if model_class is not None:
            processor.get_rope_index = types.MethodType(model_class.get_rope_index, processor)
            if hasattr(model_class, "get_vision_position_ids"):
                processor.get_vision_position_ids = types.MethodType(model_class.get_vision_position_ids, processor)
    except Exception as exc:
        warnings.warn(f"Failed to create processor: {exc}. This may affect multimodal processing", stacklevel=2)
        return None

    if "Processor" not in processor.__class__.__name__:
        return None
    return processor


def _apply_chat_template(model_config: Mapping[str, Any], tokenizer: Any, processor: Any | None) -> None:
    if (
        processor is not None
        and not getattr(processor, "chat_template", None)
        and getattr(tokenizer, "chat_template", None)
    ):
        processor.chat_template = tokenizer.chat_template

    custom_chat_template = model_config.get("custom_chat_template")
    if custom_chat_template is not None:
        if not isinstance(custom_chat_template, str):
            raise TypeError("model_config.custom_chat_template must be a string or null")
        tokenizer.chat_template = custom_chat_template
        if processor is not None:
            processor.chat_template = custom_chat_template


def load_tokenizer_from_model_config(
    model_config: Mapping[str, Any],
    *,
    source_resolver: Callable[[str], str] | None = None,
) -> Any:
    """Load only a Hugging Face tokenizer from a wire model-config mapping."""

    tokenizer_source, trust_remote_code = _resolve_tokenizer_source(model_config, source_resolver)
    tokenizer = _load_tokenizer(tokenizer_source, trust_remote_code)
    _apply_chat_template(model_config, tokenizer, None)
    return tokenizer


def load_tokenizer_and_processor_from_model_config(
    model_config: Mapping[str, Any],
    *,
    source_resolver: Callable[[str], str] | None = None,
) -> tuple[Any, Any | None]:
    """Load the Proxy tokenizer and optional multimodal processor.

    This mirrors verl's tokenizer/processor initialization without constructing
    ``HFModelConfig`` or importing verl. ``source_resolver`` may materialize a
    remote tokenizer artifact before Transformers loads both objects from it.
    """

    tokenizer_source, trust_remote_code = _resolve_tokenizer_source(model_config, source_resolver)
    tokenizer = _load_tokenizer(tokenizer_source, trust_remote_code)
    processor = _load_processor(tokenizer_source, trust_remote_code)
    _apply_chat_template(model_config, tokenizer, processor)

    return tokenizer, processor
