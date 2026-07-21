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

import base64
import io
from collections import deque

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from agent_service import AgentExecutor, TaskId, TaskSnapshot, TaskStatus
from agent_service.verl_adapter.rollout_adapter import RolloutAdapter
from verl import DataProto


class _FakeTransportClient:
    def __init__(self):
        self.submitted = []
        self.wait_responses = deque()

    def submit(self, task_spec, idempotency_key=None):
        task_id = TaskId(f"task-{len(self.submitted)}")
        self.submitted.append(task_spec)
        return task_id

    def get_status(self, task_id):
        raise AssertionError("get_status should not be used by batch completion")

    def wait_any(self, task_ids, timeout_seconds, max_results):
        return self.wait_responses.popleft()

    def cancel(self, task_id):
        return TaskSnapshot(task_id=task_id, status=TaskStatus.CANCELLED)

    def close(self):
        pass


class _Tokenizer:
    pad_token_id = 0

    @staticmethod
    def decode(token_ids, skip_special_tokens=True):
        return " ".join(map(str, token_ids))


def _object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _config():
    return OmegaConf.create(
        {
            "agent_service": {
                "wait_timeout_seconds": 5,
                "task": {
                    "problem": {},
                    "agent": {
                        "artifact": "./agent",
                        "frontend_protocol": "openai_chat_completions",
                    },
                    "execution": {"command": ["python", "main.py"]},
                    "environment": None,
                    "reward": {"reward_function": {"kind": "binary"}},
                    "generation": {},
                    "lifecycle": None,
                },
            },
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": -1,
                    "prompt_length": 4,
                    "response_length": 5,
                    "val_kwargs": {"temperature": 0, "top_p": 1.0, "top_k": -1},
                }
            },
        }
    )


def _snapshot(task_id, prompt_ids, response_ids, reward):
    return TaskSnapshot(
        task_id=TaskId(task_id),
        status=TaskStatus.SUCCEEDED,
        trajectory={
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_mask": [1] * len(response_ids),
            "response_logprobs": [-0.1] * len(response_ids),
            "num_turns": 2,
            "metrics": {"generate_sequences": 0.5},
            "extra_fields": {"trace_id": f"trace-{task_id}"},
        },
        reward={"final_reward": reward, "acc": reward},
    )


def test_adapter_restores_input_order_and_builds_training_dataproto():
    transport = _FakeTransportClient()
    executor = AgentExecutor(
        transport,
        wait_any_poll_timeout_seconds=1,
    )
    adapter = RolloutAdapter(executor, config=_config(), tokenizer=_Tokenizer())
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[0, 0, 10, 11], [0, 0, 20, 21]])},
        non_tensors={
            "raw_prompt": _object_array(
                [
                    [{"role": "user", "content": "first"}],
                    [{"role": "user", "content": "second"}],
                ]
            ),
            "uid": np.array(["uid-0", "uid-1"], dtype=object),
            "data_source": np.array(["source", "source"], dtype=object),
            "reward_model": _object_array(
                [
                    {"ground_truth": "a"},
                    {"ground_truth": "b"},
                ]
            ),
            "tools_kwargs": _object_array([{"search": {"limit": 5}}, {"search": {"limit": 7}}]),
            "custom_dataset_field": _object_array([{"nested": [1, 2]}, {"nested": [3, 4]}]),
            "__do_sample__": np.array([True, True]),
        },
    )
    transport.wait_responses.append(
        [
            _snapshot("task-1", [20, 21], [201], 0.0),
            _snapshot("task-0", [10, 11], [101, 102], 1.0),
        ]
    )

    indexed_task_ids = adapter.submit_batch(batch)
    output = adapter.wait_batch(indexed_task_ids, timeout=5)

    assert output.batch["responses"][0].tolist() == [101, 102, 0, 0, 0]
    assert output.batch["responses"][1].tolist() == [201, 0, 0, 0, 0]
    assert output.batch["rm_scores"].sum(dim=-1).tolist() == [1.0, 0.0]
    assert output.non_tensor_batch["uid"].tolist() == ["uid-0", "uid-1"]
    assert output.non_tensor_batch["acc"].tolist() == [1.0, 0.0]
    assert output.meta_info["reward_extra_keys"] == ["acc"]
    assert not hasattr(adapter, "generate_sequences")
    assert transport.submitted[0].problem["messages"][0]["content"] == "first"
    assert "raw_prompt" not in transport.submitted[0].sample_fields
    assert transport.submitted[0].sample_fields["tools_kwargs"] == {"search": {"limit": 5}}
    assert transport.submitted[1].sample_fields["custom_dataset_field"] == {"nested": [3, 4]}
    assert "__do_sample__" not in transport.submitted[0].sample_fields
    assert transport.submitted[1].reward["verifier"]["ground_truth"] == "b"
    assert transport.submitted[0].generation["max_new_tokens"] == 5
    assert adapter.in_flight == set()


def test_adapter_keeps_multimodal_payload_inline_in_messages():
    transport = _FakeTransportClient()
    executor = AgentExecutor(transport, wait_any_poll_timeout_seconds=1)
    adapter = RolloutAdapter(executor, config=_config(), tokenizer=_Tokenizer())

    image = Image.new("RGB", (3, 2), color="red")
    encoded_image = io.BytesIO()
    image.save(encoded_image, format="PNG")
    image_bytes = encoded_image.getvalue()
    raw_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "bytes": image_bytes,
                    "path": "/dataset/example.png",
                    "image": image,
                },
                {"type": "text", "text": "what is shown?"},
            ],
        }
    ]
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[0, 10, 11, 12]])},
        non_tensors={
            "raw_prompt": _object_array([raw_prompt]),
            "origin_multi_modal_data": _object_array([{"image": [image]}]),
        },
    )

    task_spec = adapter.build_task_spec(batch[0])
    payload = task_spec.to_dict()

    message_image = payload["problem"]["messages"][0]["content"][0]
    assert message_image["type"] == "image"
    assert message_image["image"]["kind"] == "image"
    assert message_image["image"]["encoding"] == "base64"
    assert message_image["image"]["content_type"] == "image/png"
    assert "raw_prompt" not in payload["sample_fields"]
    side_channel_image = payload["sample_fields"]["origin_multi_modal_data"]["image"][0]
    assert side_channel_image["kind"] == "image"
    assert side_channel_image["encoding"] == "base64"
    assert base64.b64decode(message_image["image"]["data"]) == image_bytes
    assert message_image["image"]["metadata"]["filename"] == "example.png"


def test_sample_raw_prompt_overrides_static_problem_messages():
    config = _config()
    config.agent_service.task.problem.messages = [{"role": "user", "content": "stale"}]
    adapter = RolloutAdapter(
        AgentExecutor(_FakeTransportClient(), wait_any_poll_timeout_seconds=1),
        config=config,
        tokenizer=_Tokenizer(),
    )
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[0, 10, 11, 12]])},
        non_tensors={"raw_prompt": _object_array([[{"role": "user", "content": "sample"}]])},
    )

    task_spec = adapter.build_task_spec(batch[0])

    assert task_spec.problem["messages"] == [{"role": "user", "content": "sample"}]


def test_sample_agent_name_overrides_static_agent_name_like_legacy_agent_loop():
    config = _config()
    config.agent_service.task.agent.name = "config-default-agent"
    adapter = RolloutAdapter(
        AgentExecutor(_FakeTransportClient(), wait_any_poll_timeout_seconds=1),
        config=config,
        tokenizer=_Tokenizer(),
    )
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[0, 10, 11, 12]])},
        non_tensors={
            "raw_prompt": _object_array([[{"role": "user", "content": "sample"}]]),
            "agent_name": np.array(["sample-agent"], dtype=object),
        },
    )

    task_spec = adapter.build_task_spec(batch[0])

    assert task_spec.agent["name"] == "sample-agent"
    assert task_spec.sample_fields["agent_name"] == "sample-agent"


def test_static_agent_name_is_default_when_sample_has_no_agent_name():
    config = _config()
    config.agent_service.task.agent.name = "config-default-agent"
    adapter = RolloutAdapter(
        AgentExecutor(_FakeTransportClient(), wait_any_poll_timeout_seconds=1),
        config=config,
        tokenizer=_Tokenizer(),
    )
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[0, 10, 11, 12]])},
        non_tensors={"raw_prompt": _object_array([[{"role": "user", "content": "sample"}]])},
    )

    task_spec = adapter.build_task_spec(batch[0])

    assert task_spec.agent["name"] == "config-default-agent"


def test_explicit_null_sample_agent_name_does_not_fall_back_to_static_default():
    config = _config()
    config.agent_service.task.agent.name = "config-default-agent"
    adapter = RolloutAdapter(
        AgentExecutor(_FakeTransportClient(), wait_any_poll_timeout_seconds=1),
        config=config,
        tokenizer=_Tokenizer(),
    )
    batch = DataProto.from_dict(
        tensors={"prompts": torch.tensor([[0, 10, 11, 12]])},
        non_tensors={
            "raw_prompt": _object_array([[{"role": "user", "content": "sample"}]]),
            "agent_name": np.array([None], dtype=object),
        },
    )

    task_spec = adapter.build_task_spec(batch[0])

    assert task_spec.agent["name"] is None
