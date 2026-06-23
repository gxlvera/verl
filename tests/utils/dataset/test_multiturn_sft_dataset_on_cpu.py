# Copyright 2024 Bytedance Ltd. and/or its affiliates

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
Test the MultiTurnSFTDataset implementation
"""

import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from tensordict import TensorDict
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoProcessor, AutoTokenizer
from transformers.utils import get_json_schema

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.dataset_utils import DatasetPadMode, SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.model import extract_multi_modal_inputs

custom_model_prefix = Path("~/models").expanduser().resolve()


SFT_CT_TEXT_FAMILY_CASES = [
    ("qwen25", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3", "Qwen/Qwen3-0.6B"),
    ("qwen35", "Qwen/Qwen3.5-0.8B"),
    ("minimax", "MiniMaxAI/MiniMax-Text-01"),
    ("minimaxm2", "MiniMaxAI/MiniMax-M2"),
    ("minimaxm25", "MiniMaxAI/MiniMax-M2.5"),
    ("minimaxm27", "MiniMaxAI/MiniMax-M2.7"),
    ("glm47", "zai-org/GLM-4.7-Flash"),
    ("glm5", "THUDM/GLM-5-9B-Chat"),
    ("gemma4", "google/gemma-4-27b-it"),
    ("gptoss", "openai/gpt-oss-20b"),
    ("deepseek", "deepseek-ai/DeepSeek-V3-0324"),
]

SFT_CT_TEXT_TOOL_FAMILY_CASES = [
    ("qwen25", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen3", "Qwen/Qwen3-0.6B"),
    ("minimaxm2", "MiniMaxAI/MiniMax-M2"),
    ("minimaxm25", "MiniMaxAI/MiniMax-M2.5"),
    ("glm47", "zai-org/GLM-4.7-Flash"),
    ("deepseek", "deepseek-ai/DeepSeek-V3-0324"),
]

SFT_CT_VL_FAMILY_CASES = [
    ("qwen2vl", "Qwen/Qwen2-VL-72B-Instruct", True),
    ("qwen25vl", "Qwen/Qwen2.5-VL-3B-Instruct", True),
    ("qwen3vl", "Qwen/Qwen3-VL-2B-Instruct", True),
    ("mimovl", "XiaomiMiMo/MiMo-VL-7B", False),
    ("kimivl", "moonshotai/Kimi-VL-A3B-Instruct", False),
    ("glm4v", "zai-org/GLM-4.5V", False),
    ("deepseekvl2", "deepseek-ai/deepseek-vl2-tiny", False),
]


def _load_local_tokenizer_or_skip(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Local tokenizer for {model_id!r} is not available: {exc}")


def _load_local_processor_or_skip(model_id: str):
    try:
        return AutoProcessor.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Local processor for {model_id!r} is not available: {exc}")


def _make_text_messages():
    return [
        {"role": "user", "content": "Say alpha."},
        {"role": "assistant", "content": "alpha response."},
        {"role": "user", "content": "Say beta."},
        {"role": "assistant", "content": "beta response."},
    ]


def _weather_tool_schema():
    def get_weather(city: str):
        """Get weather for a city.

        Args:
            city: Name of the city to query.
        """
        return city

    return [get_json_schema(get_weather)]


def _make_tool_messages():
    return [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]


def _image_bytes(color: str, size: tuple[int, int] = (64, 48)) -> bytes:
    image = Image.new("RGB", size, color=color)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()


def _full_encode_reference(dataset, row: dict, renderer, tools=None):
    reference_messages = dataset._build_messages(deepcopy(row))
    candidates = [reference_messages]
    parsed_messages, parsed = dataset._messages_with_parsed_tool_arguments(reference_messages)
    if parsed:
        candidates.append(parsed_messages)
    last_exception = None
    for candidate in candidates:
        try:
            return renderer.apply_chat_template(
                candidate,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"][0]
        except Exception as exc:
            last_exception = exc
    assert last_exception is not None
    raise last_exception


class _CanonicalMockTokenizer:
    pad_token_id = 0

    @staticmethod
    def _content_to_text(content):
        if isinstance(content, str):
            return content
        text = ""
        for item in content:
            if item.get("type") == "text":
                text += item.get("text", "")
            elif item.get("type") == "image":
                text += "<image>"
        return text

    def _render(self, messages, add_generation_prompt=False):
        final_assistant_idx = next(
            (idx for idx in range(len(messages) - 1, -1, -1) if messages[idx]["role"] == "assistant"),
            -1,
        )
        parts = []
        for idx, message in enumerate(messages):
            role = message["role"]
            parts.append(f"<{role}>")
            if role == "assistant" and idx == final_assistant_idx and len(messages) > 1:
                parts.append("<think></think>")
            parts.append(self._content_to_text(message.get("content", "")))
            parts.append("</s>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    @staticmethod
    def _encode(text):
        return [ord(ch) for ch in text]

    @staticmethod
    def _tensor_dict(ids, return_offsets_mapping=False):
        result = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor(
                [[(idx, idx + 1) for idx in range(len(ids))]], dtype=torch.long
            )
        return result

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        text = self._render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text

        ids = self._encode(text)
        if return_dict:
            return self._tensor_dict(ids, return_offsets_mapping=kwargs.get("return_offsets_mapping", False))
        return ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
        ids = self._encode(text)
        return self._tensor_dict(ids, return_offsets_mapping=return_offsets_mapping)

    def decode(self, ids, skip_special_tokens=False):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(chr(int(token_id)) for token_id in ids)


class _CanonicalMockImageProcessor:
    patch_size = 14


class _CanonicalMockProcessor(_CanonicalMockTokenizer):
    def __init__(self):
        self.image_processor = _CanonicalMockImageProcessor()

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        output = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            return_tensors=return_tensors,
            **kwargs,
        )
        if not tokenize or not return_dict:
            return output

        text = self._render(messages, add_generation_prompt=add_generation_prompt)
        output["mm_token_type_ids"] = torch.zeros_like(output["input_ids"])
        if "<image>" in text:
            output["pixel_values"] = torch.ones((1, 3), dtype=torch.float)
            output["image_grid_thw"] = torch.tensor([[1, 1, 1]], dtype=torch.long)
        return output


class _NativeMaskMockTokenizer(_CanonicalMockTokenizer):
    chat_template = "{% generation %}"

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        tools=None,
        return_dict=False,
        return_tensors=None,
        **kwargs,
    ):
        output = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            return_tensors=return_tensors,
            **kwargs,
        )
        if not tokenize or not return_dict or not kwargs.get("return_assistant_tokens_mask", False):
            return output

        text = self._render(messages, add_generation_prompt=add_generation_prompt)
        assistant_masks = torch.zeros_like(output["input_ids"])
        answer_start = text.find("Answer")
        if answer_start >= 0:
            assistant_masks[0, answer_start] = 1
        output["assistant_masks"] = assistant_masks
        return output


def test_multiturn_sft_continuous_token_uses_canonical_full_encode_for_text(tmp_path):
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Again"},
        {"role": "assistant", "content": "Done"},
    ]
    test_file = tmp_path / "canonical_text.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(test_file)

    tokenizer = _CanonicalMockTokenizer()
    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        processor=None,
        config={
            "max_length": 512,
            "pad_mode": "no_padding",
            "continuous_token": {"enable": True, "fallback_to_legacy": False},
        },
    )
    item = dataset[0]

    full_ids = torch.tensor(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False), dtype=torch.long
    )
    assert torch.equal(item["input_ids"], full_ids)
    assert "<think></think>Done" in tokenizer.decode(item["input_ids"])
    assert tokenizer.decode(item["input_ids"][item["loss_mask"] == 1]) == "Answer</s>Done</s>"
    assert item["loss_mask"].shape == item["input_ids"].shape == item["position_ids"].shape


def test_multiturn_sft_continuous_token_prefers_native_assistant_mask(tmp_path):
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    test_file = tmp_path / "native_mask.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(test_file)

    tokenizer = _NativeMaskMockTokenizer()
    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        processor=None,
        config={
            "max_length": 512,
            "pad_mode": "no_padding",
            "continuous_token": {"enable": True, "fallback_to_legacy": False},
        },
    )
    item = dataset[0]

    assert tokenizer.decode(item["input_ids"][item["loss_mask"] == 1]) == "A"


def test_multiturn_sft_continuous_token_uses_full_processor_encode_for_multimodal(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "mock-image"},
                {"type": "text", "text": "Describe it."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A red square."}]},
    ]
    test_file = tmp_path / "canonical_vl.parquet"
    pd.DataFrame({"messages": [messages], "tools": [[]]}).to_parquet(test_file)

    processor = _CanonicalMockProcessor()
    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=processor,
        processor=processor,
        config={
            "max_length": 512,
            "pad_mode": "no_padding",
            "continuous_token": {"enable": True, "fallback_to_legacy": False},
        },
    )
    item = dataset[0]
    decoded = processor.decode(item["input_ids"])

    assert "<image>Describe it." in decoded
    assert processor.decode(item["input_ids"][item["loss_mask"] == 1]) == "A red square.</s>"
    assert "<image>" in processor.decode(item["input_ids"][item["loss_mask"] == 0])
    assert "multi_modal_inputs" in item
    assert "pixel_values" in item["multi_modal_inputs"]
    assert "image_grid_thw" in item["multi_modal_inputs"]
    assert "mm_token_type_ids" not in item["multi_modal_inputs"]


@pytest.mark.parametrize(
    "family, model_id",
    SFT_CT_TEXT_FAMILY_CASES,
    ids=[case[0] for case in SFT_CT_TEXT_FAMILY_CASES],
)
def test_multiturn_sft_continuous_token_local_text_family_matrix(family, model_id, tmp_path):
    tokenizer = _load_local_tokenizer_or_skip(model_id)
    messages = _make_text_messages()
    test_file = tmp_path / f"{family}_text.parquet"
    pd.DataFrame({"messages": [messages]}).to_parquet(test_file)

    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        processor=None,
        config={
            "max_length": 4096,
            "pad_mode": "no_padding",
            "truncation": "error",
            "continuous_token": {"enable": True, "fallback_to_legacy": False},
        },
    )
    item = dataset[0]
    row = pd.read_parquet(test_file).iloc[0].to_dict()
    full_ids = _full_encode_reference(dataset, row, tokenizer)
    masked_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1], skip_special_tokens=False)

    assert torch.equal(item["input_ids"], full_ids)
    assert "alpha" in masked_text
    assert "beta" in masked_text
    assert item["loss_mask"].shape == item["input_ids"].shape == item["position_ids"].shape


@pytest.mark.parametrize(
    "family, model_id",
    SFT_CT_TEXT_TOOL_FAMILY_CASES,
    ids=[case[0] for case in SFT_CT_TEXT_TOOL_FAMILY_CASES],
)
def test_multiturn_sft_continuous_token_local_tool_call_family_matrix(family, model_id, tmp_path):
    tokenizer = _load_local_tokenizer_or_skip(model_id)
    tools = _weather_tool_schema()
    messages = _make_tool_messages()
    test_file = tmp_path / f"{family}_tool.parquet"
    pd.DataFrame({"messages": [messages], "tools": [tools]}).to_parquet(test_file)

    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        processor=None,
        config={
            "max_length": 4096,
            "pad_mode": "no_padding",
            "truncation": "error",
            "continuous_token": {"enable": True, "fallback_to_legacy": False},
        },
    )
    item = dataset[0]
    row = pd.read_parquet(test_file).iloc[0].to_dict()
    full_ids = _full_encode_reference(dataset, row, tokenizer, tools=tools)
    masked_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1], skip_special_tokens=False)

    assert torch.equal(item["input_ids"], full_ids)
    assert "get_weather" in masked_text
    assert "Paris" in masked_text
    assert "sunny" in masked_text


@pytest.mark.parametrize(
    "family, model_id, supported",
    SFT_CT_VL_FAMILY_CASES,
    ids=[case[0] for case in SFT_CT_VL_FAMILY_CASES],
)
def test_multiturn_sft_continuous_token_local_vl_image_family_matrix(family, model_id, supported, tmp_path):
    if not supported:
        pytest.xfail(f"{family} SFT canonical CT adapter is not implemented/verified yet")
    pytest.importorskip("qwen_vl_utils")
    tokenizer = _load_local_tokenizer_or_skip(model_id)
    processor = _load_local_processor_or_skip(model_id)
    messages = [
        {"role": "user", "content": "<image>Describe this image."},
        {"role": "assistant", "content": "The image is a red rectangle."},
        {"role": "user", "content": "What color is it?"},
        {"role": "assistant", "content": "It is red."},
    ]
    test_file = tmp_path / f"{family}_vl.parquet"
    pd.DataFrame(
        {
            "messages": [messages],
            "images": [[{"bytes": _image_bytes("red")}]],
            "tools": [[]],
        }
    ).to_parquet(test_file)

    dataset = MultiTurnSFTDataset(
        parquet_files=str(test_file),
        tokenizer=tokenizer,
        processor=processor,
        config={
            "max_length": 4096,
            "pad_mode": "no_padding",
            "truncation": "error",
            "continuous_token": {"enable": True, "fallback_to_legacy": False},
        },
    )
    item = dataset[0]
    row = pd.read_parquet(test_file).iloc[0].to_dict()
    full_ids = _full_encode_reference(dataset, row, processor)
    masked_text = tokenizer.decode(item["input_ids"][item["loss_mask"] == 1], skip_special_tokens=False)
    multi_modal_inputs = item["multi_modal_inputs"]
    image_grid_thw = multi_modal_inputs["image_grid_thw"]
    image_token_id = processor.image_token_id
    merge_size = processor.image_processor.merge_size
    expected_image_tokens = int(image_grid_thw.prod(dim=1).sum().item() // (merge_size**2))
    actual_image_tokens = int((item["input_ids"] == image_token_id).sum().item())
    masked_image_tokens = int(((item["input_ids"] == image_token_id) & (item["loss_mask"] == 1)).sum().item())

    assert torch.equal(item["input_ids"], full_ids)
    assert "red rectangle" in masked_text
    assert "It is red" in masked_text
    assert "pixel_values" in multi_modal_inputs
    assert actual_image_tokens == expected_image_tokens
    assert masked_image_tokens == 0
    assert item["position_ids"].shape[0] == 4


@pytest.mark.parametrize(
    "model_path, ignore_input_ids_mismatch",
    [
        (f"{custom_model_prefix}/Qwen/Qwen2.5-0.5B", False),
        (f"{custom_model_prefix}/Qwen/Qwen3-0.6B", True),
        (f"{custom_model_prefix}/Qwen/Qwen3.5-0.8B", False),
    ],
)
def test_multiturn_sft_dataset(model_path: str, ignore_input_ids_mismatch: bool):
    print(f"Starting test... model_path={model_path}, ignore_input_ids_mismatch={ignore_input_ids_mismatch}")
    # Create a temporary parquet file with test data
    test_data = {
        "messages": [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
                {"role": "tool", "content": "And what is 4+4?"},
                {"role": "assistant", "content": "4+4 equals 8."},
            ],
            [
                # {"role": "system", "content": "You are a powerful assistant."},
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "tool", "content": "Why?"},
                {"role": "assistant", "content": "To get to the other side!"},
            ],
        ]
    }

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test.parquet"

    # Save test data to parquet
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)

    # Initialize tokenizer and dataset
    tokenizer = hf_tokenizer(model_path)
    # processor = hf_processor(model_path)
    processor = None
    config = {
        "max_length": 512,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "ignore_input_ids_mismatch": ignore_input_ids_mismatch,
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, processor=processor, config=config)

    # Test 1: Dataset Length
    assert len(dataset) == 2, f"Expected dataset length 2, got {len(dataset)}"

    # Get items for testing
    item0 = dataset[0]  # Math conversation
    item1 = dataset[1]  # Joke conversation

    # Test 2: Required Keys and Types
    required_keys = ["input_ids", "attention_mask", "position_ids", "loss_mask"]
    for key in required_keys:
        assert key in item0, f"Missing key {key} in dataset item"
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key}"
        assert item0[key].dtype == torch.long, f"Expected torch.long for {key}, got {item0[key].dtype}"

    # Test 3: Shape Consistency
    assert item0["loss_mask"].shape == item0["input_ids"].shape, "Loss mask shape doesn't match input_ids shape"
    assert item0["attention_mask"].shape == item0["input_ids"].shape, (
        "Attention mask shape doesn't match input_ids shape"
    )
    assert item0["position_ids"].shape == item0["input_ids"].shape, "Position IDs shape doesn't match input_ids shape"

    # Test 4: Loss Mask Pattern - Math Conversation
    loss_mask0 = item0["loss_mask"]
    input_ids0 = item0["input_ids"]

    # Find assistant response positions
    assistant_positions0 = torch.where(loss_mask0 == 1)[0]
    assert len(assistant_positions0) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    assistant_text0 = tokenizer.decode(input_ids0[loss_mask0 == 1])
    print(f"Math conversation assistant text: {assistant_text0}")
    assert "2+2 equals 4" in assistant_text0, "First assistant response not found"
    assert "4+4 equals 8" in assistant_text0, "Second assistant response not found"

    # Test 5: Loss Mask Pattern - Joke Conversation
    loss_mask1 = item1["loss_mask"]
    input_ids1 = item1["input_ids"]

    # Find assistant response positions
    assistant_positions1 = torch.where(loss_mask1 == 1)[0]
    assert len(assistant_positions1) > 0, "No assistant positions found in loss mask"

    # Decode and verify assistant responses
    assistant_text1 = tokenizer.decode(input_ids1[loss_mask1 == 1])
    print(f"Joke conversation assistant text: {assistant_text1}")
    assert "chicken cross the road" in assistant_text1, "First assistant response not found"
    assert "other side" in assistant_text1, "Second assistant response not found"

    # Test 6: Attention Mask Pattern
    attention_mask0 = item0["attention_mask"]
    sequence_length = torch.sum(attention_mask0)
    assert sequence_length > 0, "No tokens marked as attended in attention mask"
    assert torch.all(attention_mask0[:sequence_length] == 1), "Incorrect attention mask pattern"
    if sequence_length < len(attention_mask0):
        assert torch.all(attention_mask0[sequence_length:] == 0), "Padding not properly masked"

    # Test 7: Position IDs Pattern
    position_ids0 = item0["position_ids"]
    assert torch.equal(position_ids0[:sequence_length], torch.arange(sequence_length)), (
        "Position IDs not sequential for non-padded tokens"
    )
    if sequence_length < len(position_ids0):
        assert torch.all(position_ids0[sequence_length:] == 0), "Padding position IDs not zero"

    # Test 8: Verify loss mask for assistant responses
    # Get the full conversation text
    full_text = tokenizer.decode(input_ids0)
    print(f"\nFull conversation text:\n{full_text}")

    # Get the assistant responses
    assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 1])
    print(f"\nAssistant responses (from loss mask):\n{assistant_text}")

    # Verify that loss mask is set for all assistant responses
    for msg in test_data["messages"][0]:  # First conversation
        if msg["role"] == "assistant":
            # The content should appear in the masked text
            assert msg["content"] in assistant_text, f"Assistant message '{msg['content']}' not found in masked text"

            # The content should NOT appear in the non-masked text
            non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
            assert msg["content"] not in non_assistant_text, (
                f"Assistant message '{msg['content']}' found in non-assistant text"
            )

    # Test 9: Verify non-assistant parts have loss_mask=0
    # Get non-assistant text
    non_assistant_text = tokenizer.decode(input_ids0[loss_mask0 == 0])
    print(f"\nNon-assistant text (from loss mask):\n{non_assistant_text}")

    # Verify that system and user messages are in the non-assistant text
    for msg in test_data["messages"][0]:  # First conversation
        if msg["role"] in ["system", "user"]:
            assert msg["content"] in non_assistant_text, (
                f"{msg['role'].title()} message '{msg['content']}' not found in non-assistant text"
            )

            # And verify they're NOT in the assistant text
            assert msg["content"] not in assistant_text, (
                f"{msg['role'].title()} message '{msg['content']}' found in assistant text"
            )

    # Test 10: Verify padding behavior
    padding_config = {
        "max_length": 1024,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "ignore_input_ids_mismatch": ignore_input_ids_mismatch,
    }
    small_dataset = MultiTurnSFTDataset(
        parquet_files=test_file, tokenizer=tokenizer, processor=processor, config=padding_config
    )
    padded_item = small_dataset[0]

    # Get actual sequence length (before padding)
    actual_length = torch.sum(padded_item["attention_mask"])

    # Verify padding tokens
    assert torch.all(padded_item["input_ids"][actual_length:] == tokenizer.pad_token_id), (
        "Padding tokens not set correctly"
    )
    assert torch.all(padded_item["attention_mask"][actual_length:] == 0), "Attention mask not set correctly for padding"
    assert torch.all(padded_item["loss_mask"][actual_length:] == 0), "Loss mask not set correctly for padding"

    # test no-padding
    config = {
        "max_length": 512,
        "truncation": "error",
        "multiturn": {"messages_key": "messages"},
        "pad_mode": "no_padding",
        "ignore_input_ids_mismatch": ignore_input_ids_mismatch,
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, processor=processor, config=config)

    item0 = dataset[0]

    # Verify that the output contains expected keys for no-padding mode
    required_keys = ["input_ids", "position_ids", "loss_mask"]
    for key in required_keys:
        assert key in item0, f"Missing key {key} in no-padding mode dataset item"
        assert isinstance(item0[key], torch.Tensor), f"Expected torch.Tensor for {key} in no-padding mode"

    # make sure assistant_text matches with expected
    assistant_text = tokenizer.decode(item0["input_ids"][item0["loss_mask"] == 1])
    assert assistant_text == "2+2 equals 4.<|im_end|>\n4+4 equals 8.<|im_end|>\n"

    print("All tests passed!")
    print("Starting test...")


@pytest.mark.parametrize(
    "model_path, apply_chat_template_kwargs",
    [
        (f"{custom_model_prefix}/openai/gpt-oss-20b", {"model_identity": "You are a helpful assistant."}),
    ],
)
def test_multiturn_sft_dataset_with_chat_template_kwargs(model_path: str, apply_chat_template_kwargs: dict):
    """Test that custom apply_chat_template_kwargs are forwarded to system prompt
    measurement so the loss mask is not shifted when kwargs change tokenization.

    Some chat templates embed configurable fields (e.g. model_identity) in the
    system prompt. If these kwargs are not forwarded to system prompt length
    measurement, the per-turn strip length is wrong, causing role markers to be
    removed and the loss mask to shift.
    """
    test_data = {
        "messages": [
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2+2 equals 4."},
            ],
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
            ],
        ]
    }

    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test_kwargs.parquet"
    df = pd.DataFrame(test_data)
    df.to_parquet(test_file)

    tokenizer = hf_tokenizer(model_path)

    config = {
        "max_length": 1024,
        "truncation": "error",
        "pad_mode": "no_padding",
        "apply_chat_template_kwargs": apply_chat_template_kwargs,
    }
    dataset = MultiTurnSFTDataset(parquet_files=test_file, tokenizer=tokenizer, processor=None, config=config)

    for idx in range(len(dataset)):
        item = dataset[idx]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]

        assistant_text = tokenizer.decode(input_ids[loss_mask == 1])
        non_assistant_text = tokenizer.decode(input_ids[loss_mask == 0])

        for msg in test_data["messages"][idx]:
            if msg["role"] == "assistant":
                assert msg["content"] in assistant_text, (
                    f"Assistant message '{msg['content']}' not found in masked text. "
                    f"This may indicate system prompt length mismatch when using "
                    f"custom chat_template_kwargs."
                )
                assert msg["content"] not in non_assistant_text, (
                    f"Assistant message '{msg['content']}' found in non-assistant text"
                )
            elif msg["role"] in ["system", "user"]:
                assert msg["content"] in non_assistant_text, (
                    f"{msg['role'].title()} message '{msg['content']}' not found in non-assistant text"
                )
                assert msg["content"] not in assistant_text, (
                    f"{msg['role'].title()} message '{msg['content']}' found in assistant text"
                )

    print("All chat_template_kwargs tests passed!")


def generate_image(description: str, size: str = "256x256"):
    """Generate a simple image based on description.

    Args:
        description: The description of the image to generate.
        size: The size of the image. Defaults to "256x256". (choices: ["256x256", "512x512"])

    Returns:
        A generated image
    """
    ...


@pytest.fixture
def vlm_data_file():
    test_data = [
        # sample 0: single turn with image input
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>Describe this image.",
                },
                {
                    "role": "assistant",
                    "content": "The image is a red square.",
                },
            ],
            "images": [Image.new("RGB", (300, 300), color="red")],
            "tools": [],
        },
        # sample 1: single turn with multiple images input
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image><image>Compare these images.",
                },
                {
                    "role": "assistant",
                    "content": "The first image is a red square and the second image is a green square.",
                },
            ],
            "images": [Image.new("RGB", (100, 100), color="red"), Image.new("RGB", (100, 300), color="green")],
            "tools": [],
        },
        # sample 2: multi turn with image input and tool generated image
        {
            "messages": [
                {
                    "role": "user",
                    "content": "<image>Describe this image.",
                },
                {
                    "role": "assistant",
                    "content": "Let's generate a zoom-in image.",
                    "tool_calls": [
                        {
                            "function": {"arguments": {"bbox_2d": "[0, 1, 2, 4]"}, "name": "image_zoom_in_tool"},
                            "type": "function",
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "<image>Generated image.",
                },
                {"role": "assistant", "content": "The zoom-in image is a red square."},
            ],
            "images": [Image.new("RGB", (300, 500), color="red"), Image.new("RGB", (100, 100), color="red")],
            "tools": [get_json_schema(generate_image)],
        },
        # sample 3: single turn without image input
        {
            "messages": [
                {"role": "user", "content": "How is the weather today?"},
                {"role": "assistant", "content": "The weather is sunny."},
            ],
            "images": [],
            "tools": [],
        },
    ]

    # Create test directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    test_file = "test_data/test_vlm.parquet"

    # Save test data to parquet
    df = pd.DataFrame(test_data)

    def serialize_image(img):
        if isinstance(img, Image.Image):
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            return {"bytes": img_byte_arr.getvalue()}
        return img

    df["images"] = df["images"].apply(lambda x: [serialize_image(img) for img in x])

    df.to_parquet(test_file)
    return test_file


@pytest.mark.parametrize(
    "model_path",
    [
        f"{custom_model_prefix}/Qwen/Qwen3-VL-2B-Instruct",
        f"{custom_model_prefix}/Qwen/Qwen3.5-0.8B",
    ],
)
def test_multiturn_sft_vlm_dataset_on_cpu(model_path, vlm_data_file):
    df = pd.read_parquet(vlm_data_file)
    tokenizer = hf_tokenizer(model_path)
    processor = hf_processor(model_path)
    config = {"max_length": 1024, "pad_mode": "no_padding", "truncation": "error", "messages_key": "messages"}
    dataset = MultiTurnSFTDataset(parquet_files=vlm_data_file, tokenizer=tokenizer, processor=processor, config=config)
    assert dataset.pad_mode == DatasetPadMode.NO_PADDING

    for i in range(len(dataset)):
        item = dataset[i]
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        position_ids = item["position_ids"]
        pixel_values = item.get("multi_modal_inputs", {}).get("pixel_values")
        image_grid_thw = item.get("multi_modal_inputs", {}).get("image_grid_thw")

        assert input_ids.shape == loss_mask.shape, "Shapes of input_ids and loss_mask must be equal"
        assert position_ids.dim() == 2, "position_ids must be 2-dimensional"
        assert position_ids.shape[0] == 4, f"position_ids[0] should be 4: {position_ids[0]}"
        assert position_ids.shape[1] == input_ids.shape[0]

        # 1. verify input_ids without assistant text
        text = tokenizer.decode(input_ids[loss_mask == 0], skip_special_tokens=True)
        print(f"Text without assistant: {repr(text)}")
        for message in df["messages"][i]:
            if message["role"] != "assistant":
                content = message["content"].replace("<image>", "")
                assert content in text, f"user/tool text should be in the input_ids: {text}"

        # 2. verify input_ids with assistant text
        text = tokenizer.decode(input_ids[loss_mask == 1], skip_special_tokens=True)
        print(f"Text with assistant: {repr(text)}")
        for message in df["messages"][i]:
            if message["role"] == "assistant":
                assert message["content"] in text, f"Assistant text should be in the input_ids: {text}"
                assert "assistant" not in text, f"Assistant token should not be in the input_ids: {text}"

        # 3. verify image token match with image_grid_thw
        if len(df["images"][i]) > 0:
            patch_size = processor.image_processor.patch_size
            temporal_patch_size = processor.image_processor.temporal_patch_size
            merge_size = processor.image_processor.merge_size
            num_patches = image_grid_thw.prod(dim=1).sum()
            assert image_grid_thw.shape == (len(df["images"][i]), 3), (
                f"image_grid_thw: {image_grid_thw.shape} should have shape ({len(df['images'][i])}, 3)"
            )
            assert pixel_values.shape == (num_patches, 3 * temporal_patch_size * patch_size * patch_size), (
                f"pixel_values: {pixel_values.shape} should have shape ({num_patches}, {3 * patch_size * patch_size})"
            )
            assert (input_ids == processor.image_token_id).sum() == num_patches // (merge_size**2)
        else:
            assert pixel_values is None, "pixel_values should be None when no image is provided"
            assert image_grid_thw is None, "image_grid_thw should be None when no image is provided"


@pytest.mark.parametrize(
    "model_path",
    [
        f"{custom_model_prefix}/Qwen/Qwen3-VL-2B-Instruct",
    ],
)
def test_multiturn_sft_vlm_dataloader_on_cpu(model_path, vlm_data_file):
    df = pd.read_parquet(vlm_data_file)
    tokenizer = hf_tokenizer(model_path)
    processor = hf_processor(model_path)
    config = {"max_length": 1024, "pad_mode": "no_padding", "truncation": "error", "messages_key": "messages"}
    dataset = MultiTurnSFTDataset(parquet_files=vlm_data_file, tokenizer=tokenizer, processor=processor, config=config)
    assert dataset.pad_mode == DatasetPadMode.NO_PADDING

    collate_fn = SFTTensorCollator(DatasetPadMode.NO_PADDING)
    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=1, rank=0, drop_last=True)
    batch_size = 2
    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    for i, batch in enumerate(dataloader):
        # 1. verify input_ids, loss_mask
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"]
        assert input_ids.is_nested, "input_ids should be a nested tensor"
        assert loss_mask.is_nested, "loss_mask should be a nested tensor"
        assert input_ids.shape[0] == loss_mask.shape[0] == batch_size, "Shapes of input_ids, loss_mask must be equal"

        # 2. verify position_ids: (bs, 4, seq_len)
        position_ids = batch["position_ids"]
        assert position_ids.is_nested, "position_ids should be a nested tensor"
        assert position_ids.dim() == 3, "position_ids must be 3-dimensional"
        assert position_ids.shape[0] == batch_size
        values = position_ids.values()
        assert values.shape == (4, len(input_ids.values()))

        # 3. verify multi-modal data
        td = TensorDict(**batch, batch_size=batch_size)
        multi_modal_inputs = extract_multi_modal_inputs(td["multi_modal_inputs"])
        pixel_values = multi_modal_inputs["pixel_values"]
        image_grid_thw = multi_modal_inputs["image_grid_thw"]

        num_images = sum([len(images) for images in df["images"][i * batch_size : (i + 1) * batch_size]])
        assert image_grid_thw.shape == (num_images, 3), (
            f"image_grid_thw: {image_grid_thw.shape} should have shape ({num_images}, 3)"
        )
        patch_size = processor.image_processor.patch_size
        temporal_patch_size = processor.image_processor.temporal_patch_size
        num_patches = image_grid_thw.prod(dim=1).sum()
        assert pixel_values.shape[0] == num_patches, (
            f"pixel_values: {pixel_values.shape} should have shape "
            f"({num_patches}, 3 * {temporal_patch_size} * {patch_size} * {patch_size})"
        )
