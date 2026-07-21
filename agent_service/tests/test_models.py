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

from datetime import timezone

import pytest

from agent_service import AgentServiceProtocolError, TaskSnapshot, TaskSpec, TaskStatus


def test_task_spec_serializes_only_public_top_level_fields():
    task_spec = TaskSpec(
        problem={"messages": [{"role": "user", "content": "hello"}]},
        agent={"artifact": "./agent", "frontend_protocol": "openai_chat_completions"},
        execution={"command": ("python", "main.py")},
        reward={"verifier": {"kind": "answer_match"}},
        generation={"temperature": 0.7, "max_new_tokens": 128},
        lifecycle={"timeout_seconds": 60},
    )

    assert task_spec.to_dict() == {
        "problem": {"messages": [{"role": "user", "content": "hello"}]},
        "agent": {"artifact": "./agent", "frontend_protocol": "openai_chat_completions"},
        "execution": {"command": ["python", "main.py"]},
        "reward": {"verifier": {"kind": "answer_match"}},
        "generation": {"temperature": 0.7, "max_new_tokens": 128},
        "sample_fields": {},
        "lifecycle": {"timeout_seconds": 60},
    }


def test_task_spec_serializes_open_sample_fields():
    task_spec = TaskSpec(
        problem={"messages": [{"role": "user", "content": "hello"}]},
        agent={},
        execution={},
        reward={},
        generation={},
        sample_fields={"tools_kwargs": {"search": {}}, "custom_key": {"x": 1}},
    )

    payload = task_spec.to_dict()

    assert payload["sample_fields"]["custom_key"] == {"x": 1}


def test_task_spec_rejects_non_json_mapping_keys():
    task_spec = TaskSpec(
        problem={1: "not-json"},
        agent={},
        execution={},
        reward={},
        generation={},
    )

    with pytest.raises(TypeError, match="keys must be strings"):
        task_spec.to_dict()


def test_task_snapshot_parses_terminal_result_and_preserves_extensions():
    snapshot = TaskSnapshot.from_dict(
        {
            "task_id": "task-1",
            "session_id": "session-1",
            "status": "SUCCEEDED",
            "created_at": "2026-07-16T10:00:00Z",
            "completed_at": "2026-07-16T10:01:00+00:00",
            "trajectory": {"token_ids": [1, 2, 3]},
            "reward": {"final_reward": 1, "tool_reward": 0.2},
            "server_extension": "kept",
        }
    )

    assert snapshot.status is TaskStatus.SUCCEEDED
    assert snapshot.is_terminal
    assert snapshot.final_reward == 1.0
    assert snapshot.created_at.tzinfo is timezone.utc
    assert snapshot.extra == {"server_extension": "kept"}


def test_task_snapshot_rejects_unknown_status():
    with pytest.raises(AgentServiceProtocolError, match="Unknown Task status"):
        TaskSnapshot.from_dict({"task_id": "task-1", "status": "RETRYING"})
