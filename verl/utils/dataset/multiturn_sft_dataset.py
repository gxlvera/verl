# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import logging
import os
import re
from copy import deepcopy
from functools import wraps
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image, process_video
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.py_functional import convert_nested_value_to_list_recursive
from verl.utils.tokenizer.chat_template import apply_chat_template, extract_system_prompt_and_generation

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_WARNED_MESSAGES: set[str] = set()


def _warning_once(message: str):
    if message not in _WARNED_MESSAGES:
        _WARNED_MESSAGES.add(message)
        logger.warning(message)


def once(func):
    """Decorator to ensure a function runs only once. Subsequent calls do nothing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "called"):
            wrapper.called = True
            return func(*args, **kwargs)

    return wrapper


@once
def print_assembled_message(tokenizer, message_list, input_ids, loss_mask, attn_mask, tools):
    """
    Print the message after applying the chat template
    """

    tokenized = tokenizer.apply_chat_template(message_list, add_generation_prompt=False, tokenize=False, tools=tools)
    sep = "\n\n"
    str = f"tokenized entire message:\n{tokenized}"
    str += sep
    decoded_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    str += f"tokenized seperately    :\n{tokenizer.decode(decoded_ids)}"

    logger.debug(str)


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int, optional): Limit the number of samples. Defaults to -1 (use all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        self.messages_key = config.get("messages_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.enable_thinking_default = config.get("enable_thinking_default", None)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        continuous_token_config = config.get("continuous_token", {})
        self.continuous_token_enabled = continuous_token_config.get("enable", False)
        self.continuous_token_fallback_to_legacy = continuous_token_config.get("fallback_to_legacy", True)
        self.continuous_token_use_native_mask = continuous_token_config.get("use_native_mask", True)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        self._supports_native_assistant_mask = self._chat_template_has_generation_block()

        self._download()
        self._read_files_and_process()

    def _chat_template_has_generation_block(self) -> bool:
        template = getattr(self.tokenizer, "chat_template", None)
        if template is None and self.processor is not None:
            template = getattr(self.processor, "chat_template", None)
        if isinstance(template, dict):
            template = "\n".join(str(value) for value in template.values())
        return "{% generation" in str(template)

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # default loader loads some list as np.ndarray, which fails the tokenizer
            dataframe = pd.read_parquet(parquet_file, dtype_backend="pyarrow")
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        # system prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # generation prompt: <|im_start|>assistant\n
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )

    def __len__(self):
        return len(self.messages)

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process a single message and return its tokenized representation.

        Args:
            index: turn index in the conversation
            message: A single message dictionary
            images: List of images to be used
            videos: List of videos to be used
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        inputs = apply_chat_template(
            processor,
            messages=[message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            # mask out generation prompt if assistant message
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def _apply_chat_template_kwargs(self, enable_thinking: Optional[bool] = None) -> dict[str, Any]:
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        return apply_chat_template_kwargs

    def _apply_full_chat_template(
        self,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
        return_offsets_mapping: bool = False,
    ):
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = self._apply_chat_template_kwargs(enable_thinking)
        if self.continuous_token_use_native_mask and self._supports_native_assistant_mask:
            apply_chat_template_kwargs["return_assistant_tokens_mask"] = True
        if return_offsets_mapping:
            if self.processor is not None:
                processor_kwargs = dict(apply_chat_template_kwargs.get("processor_kwargs", {}))
                processor_kwargs["return_offsets_mapping"] = True
                apply_chat_template_kwargs["processor_kwargs"] = processor_kwargs
            else:
                apply_chat_template_kwargs["return_offsets_mapping"] = True
        return apply_chat_template(
            processor,
            messages=messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

    @staticmethod
    def _find_subsequence(sequence: list[int], pattern: list[int], start: int = 0) -> int:
        if not pattern:
            return -1
        max_start = len(sequence) - len(pattern)
        if max_start < start:
            return -1
        first = pattern[0]
        for i in range(start, max_start + 1):
            if sequence[i] == first and sequence[i : i + len(pattern)] == pattern:
                return i
        return -1

    @staticmethod
    def _message_text_content(message: dict[str, Any]) -> Optional[str]:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if not isinstance(item, dict):
                    return None
                item_type = item.get("type")
                if item_type == "text":
                    texts.append(item.get("text", ""))
                elif "text" in item and item_type is None:
                    texts.append(item.get("text", ""))
                else:
                    return None
            return "".join(texts)
        return None

    @staticmethod
    def _replace_message_text_content(message: dict[str, Any], text: str):
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = text
            return
        if isinstance(content, list):
            message["content"] = [{"type": "text", "text": text}]
            return
        raise ValueError("Unsupported assistant content type for canonical loss mask recovery.")

    def _render_chat_template_text(
        self,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        processor = self.processor if self.processor is not None else self.tokenizer
        return apply_chat_template(
            processor,
            messages=messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            **self._apply_chat_template_kwargs(enable_thinking),
        )

    def _get_offset_mapping(
        self,
        input_ids: torch.Tensor,
        full_text: str,
        encoded_offsets: Optional[torch.Tensor | list] = None,
    ) -> Optional[torch.Tensor]:
        if encoded_offsets is not None:
            offset_mapping = encoded_offsets
            if not isinstance(offset_mapping, torch.Tensor):
                offset_mapping = torch.tensor(offset_mapping, dtype=torch.long)
            if offset_mapping.dim() == 3:
                offset_mapping = offset_mapping[0]
            if offset_mapping.shape[0] == input_ids.shape[0]:
                return offset_mapping.to(dtype=torch.long)

        if self.processor is not None:
            return None

        try:
            tokenized = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        except Exception:
            return None
        raw_input_ids = tokenized["input_ids"][0]
        if not torch.equal(raw_input_ids, input_ids):
            return None
        return tokenized["offset_mapping"][0].to(dtype=torch.long)

    def _assistant_content_char_spans(
        self,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
        full_text: str,
    ) -> dict[int, tuple[int, int]]:
        spans = {}
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            content = self._message_text_content(message)
            if content is None or content == "":
                continue

            sentinel = f"<<<VERL_ASSISTANT_CONTENT_{index}_SENTINEL>>>"
            sentinel_messages = deepcopy(messages)
            self._replace_message_text_content(sentinel_messages[index], sentinel)
            sentinel_text = self._render_chat_template_text(sentinel_messages, tools, enable_thinking)
            sentinel_start = sentinel_text.find(sentinel)
            if sentinel_start < 0:
                continue
            content_end = sentinel_start + len(content)
            if full_text[sentinel_start:content_end] != content:
                continue
            spans[index] = (sentinel_start, content_end)
        return spans

    @staticmethod
    def _mask_offset_span(loss_mask: torch.Tensor, offset_mapping: torch.Tensor, char_span: tuple[int, int]) -> bool:
        span_start, span_end = char_span
        matched = False
        for token_index, (token_start, token_end) in enumerate(offset_mapping.tolist()):
            if token_end <= token_start:
                continue
            if token_start < span_end and token_end > span_start:
                loss_mask[token_index] = 1
                matched = True
        return matched

    def _build_canonical_loss_mask(
        self,
        input_ids: torch.Tensor,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
        full_text: str,
        offset_mapping: Optional[torch.Tensor],
    ) -> torch.Tensor:
        loss_mask = torch.zeros_like(input_ids)
        input_ids_list = input_ids.tolist()
        cursor = 0
        char_spans = (
            self._assistant_content_char_spans(messages, tools, enable_thinking, full_text)
            if offset_mapping is not None
            else {}
        )

        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue

            _input_ids, _loss_mask, _, _ = self._process_single_message(
                index=index,
                message=message,
                full_message=messages,
                tools=tools if index == 0 else None,
                enable_thinking=enable_thinking,
            )
            target_ids = _input_ids[_loss_mask == 1].tolist()
            target_start = self._find_subsequence(input_ids_list, target_ids, cursor)
            if target_start >= 0:
                target_end = target_start + len(target_ids)
                loss_mask[target_start:target_end] = 1
                cursor = target_end
                continue

            if index in char_spans and offset_mapping is not None:
                if self._mask_offset_span(loss_mask, offset_mapping, char_spans[index]):
                    masked_positions = torch.where(loss_mask == 1)[0]
                    cursor = int(masked_positions[-1].item()) + 1 if len(masked_positions) > 0 else cursor
                    continue

            raise ValueError(
                "Unable to recover canonical SFT loss_mask for an assistant turn. "
                "This sample likely needs native generation masks or a model-specific mask implementation."
            )

        return loss_mask

    @staticmethod
    def _native_assistant_loss_mask(input_ids: torch.Tensor, assistant_masks: Optional[Any]) -> Optional[torch.Tensor]:
        if assistant_masks is None:
            return None
        if not isinstance(assistant_masks, torch.Tensor):
            assistant_masks = torch.tensor(assistant_masks, dtype=torch.long)
        if assistant_masks.dim() == 2:
            assistant_masks = assistant_masks[0]
        assistant_masks = assistant_masks.to(dtype=torch.long)
        if assistant_masks.shape != input_ids.shape or assistant_masks.sum().item() == 0:
            return None
        return assistant_masks


    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", "video": video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        messages: list = convert_nested_value_to_list_recursive(example[self.messages_key])
        images = example[self.image_key] if self.image_key in example else []
        videos = example[self.video_key] if self.video_key in example else []

        image_offset, video_offset = 0, 0
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                continue

            if self.image_key not in example and self.video_key not in example:
                if self.processor is not None:
                    message["content"] = [{"type": "text", "text": content}]
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    video = process_video(videos[video_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "video", "video": video})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = (
            self.enable_thinking[item] if self.enable_thinking is not None else self.enable_thinking_default
        )
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)

        if self.continuous_token_enabled:
            try:
                return self._build_item_canonical(messages, tools, enable_thinking)
            except Exception as exc:
                if not self.continuous_token_fallback_to_legacy:
                    raise
                _warning_once(
                    "Falling back to legacy MultiTurnSFTDataset assembly because canonical continuous token "
                    f"assembly failed: {exc}"
                )

        return self._build_item_legacy(messages, tools, enable_thinking)

    def _build_item_legacy(
        self,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
    ):
        # 1. tokenize each message
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                index=i,
                message=message,
                full_message=messages,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        print_assembled_message(self.tokenizer, messages, input_ids, loss_mask, attention_mask, tools)
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        multi_modal_inputs = self._merge_legacy_multi_modal_inputs(multi_modal_inputs)
        return self._postprocess_item(input_ids, loss_mask, attention_mask, multi_modal_inputs)

    @staticmethod
    def _merge_legacy_multi_modal_inputs(multi_modal_inputs: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes
        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if k == "mm_token_type_ids":
                keys_to_remove.append(k)
                continue
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                # Check if all tensors in the list have the same shape
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]

        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        return multi_modal_inputs

    def _build_item_canonical(
        self,
        messages: list[dict],
        tools: Optional[list[dict[str, Any]]],
        enable_thinking: Optional[bool],
    ):
        try:
            inputs = dict(
                self._apply_full_chat_template(
                    messages,
                    tools=tools,
                    enable_thinking=enable_thinking,
                    return_offsets_mapping=True,
                )
            )
        except Exception:
            inputs = dict(
                self._apply_full_chat_template(
                    messages,
                    tools=tools,
                    enable_thinking=enable_thinking,
                    return_offsets_mapping=False,
                )
            )
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]
        encoded_offsets = inputs.pop("offset_mapping", None)
        assistant_masks = inputs.pop("assistant_masks", None)

        loss_mask = self._native_assistant_loss_mask(input_ids, assistant_masks)
        if loss_mask is None:
            try:
                full_text = self._render_chat_template_text(messages, tools, enable_thinking)
                offset_mapping = self._get_offset_mapping(input_ids, full_text, encoded_offsets)
            except Exception:
                full_text = ""
                offset_mapping = None
            loss_mask = self._build_canonical_loss_mask(
                input_ids=input_ids,
                messages=messages,
                tools=tools,
                enable_thinking=enable_thinking,
                full_text=full_text,
                offset_mapping=offset_mapping,
            )
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )

        multi_modal_inputs = self._canonical_multi_modal_inputs(inputs)
        return self._postprocess_item(input_ids, loss_mask, attention_mask, multi_modal_inputs)

    @staticmethod
    def _canonical_multi_modal_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        multi_modal_inputs = {}
        for key, value in inputs.items():
            if key in {"mm_token_type_ids", "assistant_masks"} or value is None:
                continue
            multi_modal_inputs[key] = value
        return multi_modal_inputs

    def _postprocess_item(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: dict[str, Any],
    ):
        # 2. handle position_ids for Qwen-VL series models
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )  # (3, seq_len)
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # 3. handle padding
        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            if sequence_length > self.max_length and self.truncation == "error":
                raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
            # truncate input_ids if it is longer than max_length
            if len(input_ids) > self.max_length:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[..., : self.max_length]

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """Check concatenated input_ids of apply_chat_template to each turn equals
        apply_chat_template to whole messages.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                _warning_once(error_message)
            else:
                raise AssertionError(error_message)
