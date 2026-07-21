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

import sys

import pytest

from agent_service.proxy.tokenizer import (
    load_tokenizer_and_processor_from_model_config,
    load_tokenizer_from_model_config,
)


class _Tokenizer:
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token_id = None
    pad_token = None
    chat_template = "original"


class _AutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return _Tokenizer()


class _PreTrainedTokenizerBase:
    pass


class MllamaProcessor:
    chat_template = None


class _AutoProcessor:
    calls = []
    result = MllamaProcessor()

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return cls.result


class _AutoConfig:
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        return {"source": source}


def test_load_tokenizer_uses_serialized_model_config_without_verl(monkeypatch):
    fake_transformers = type("Transformers", (), {"AutoTokenizer": _AutoTokenizer})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    imported_modules = []
    monkeypatch.setattr("agent_service.proxy.tokenizer.importlib.import_module", imported_modules.append)
    _AutoTokenizer.calls.clear()

    tokenizer = load_tokenizer_from_model_config(
        {
            "path": "hdfs://models/policy",
            "tokenizer_path": "hdfs://tokenizers/policy",
            "trust_remote_code": True,
            "external_lib": ["custom_tokenizer"],
            "custom_chat_template": "{{ messages }}",
        },
        source_resolver=lambda source: source.replace("hdfs://", "/local/"),
    )

    assert imported_modules == ["custom_tokenizer"]
    assert _AutoTokenizer.calls == [("/local/tokenizers/policy", {"trust_remote_code": True})]
    assert tokenizer.pad_token_id == tokenizer.eos_token_id
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.chat_template == "{{ messages }}"


def test_load_tokenizer_falls_back_to_model_path(monkeypatch):
    fake_transformers = type("Transformers", (), {"AutoTokenizer": _AutoTokenizer})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _AutoTokenizer.calls.clear()

    load_tokenizer_from_model_config({"path": "Qwen/Qwen3-8B", "tokenizer_path": None})

    assert _AutoTokenizer.calls == [("Qwen/Qwen3-8B", {"trust_remote_code": False})]


def test_load_tokenizer_and_multimodal_processor(monkeypatch):
    fake_transformers = type(
        "Transformers",
        (),
        {
            "AutoConfig": _AutoConfig,
            "AutoProcessor": _AutoProcessor,
            "AutoTokenizer": _AutoTokenizer,
            "PreTrainedTokenizerBase": _PreTrainedTokenizerBase,
        },
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _AutoTokenizer.calls.clear()
    _AutoProcessor.calls.clear()
    _AutoConfig.calls.clear()
    _AutoProcessor.result = MllamaProcessor()

    tokenizer, processor = load_tokenizer_and_processor_from_model_config(
        {
            "path": "Qwen/Qwen-VL",
            "trust_remote_code": True,
            "custom_chat_template": "{{ messages }}",
        }
    )

    expected_call = [("Qwen/Qwen-VL", {"trust_remote_code": True})]
    assert _AutoTokenizer.calls == expected_call
    assert _AutoProcessor.calls == expected_call
    assert _AutoConfig.calls == expected_call
    assert processor.config == {"source": "Qwen/Qwen-VL"}
    assert tokenizer.chat_template == "{{ messages }}"
    assert processor.chat_template == "{{ messages }}"


def test_text_only_processor_falls_back_to_none(monkeypatch):
    fake_transformers = type(
        "Transformers",
        (),
        {
            "AutoConfig": _AutoConfig,
            "AutoProcessor": _AutoProcessor,
            "AutoTokenizer": _AutoTokenizer,
            "PreTrainedTokenizerBase": _PreTrainedTokenizerBase,
        },
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    _AutoProcessor.result = _PreTrainedTokenizerBase()

    _, processor = load_tokenizer_and_processor_from_model_config({"path": "Qwen/Qwen3-8B"})

    assert processor is None


def test_load_tokenizer_requires_an_artifact_location():
    with pytest.raises(ValueError, match="tokenizer_path or model_config.path"):
        load_tokenizer_from_model_config({"path": None, "tokenizer_path": None})
