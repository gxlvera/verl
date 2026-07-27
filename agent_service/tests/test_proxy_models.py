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

from agent_service.proxy import (
    CanonicalMessage,
    GenerationSpec,
    ProxyError,
    TokenProvenance,
    Trajectory,
    TrajectoryBundle,
)


def test_generation_spec_copies_sampling_params():
    sampling_params = {"temperature": 0.7, "stop": ["done"]}
    spec = GenerationSpec(sampling_params=sampling_params)
    sampling_params["stop"].append("mutated")

    assert spec.sampling_params == {"temperature": 0.7, "stop": ["done"]}


def test_canonical_message_is_immutable_stable_json():
    source = {"content": {"b": 2, "a": 1}, "role": "user"}
    message = CanonicalMessage.from_dict(source)
    source["content"]["a"] = 999

    assert message.to_dict() == {"content": {"a": 1, "b": 2}, "role": "user"}
    assert message.canonical_json == '{"content":{"a":1,"b":2},"role":"user"}'


def test_trajectory_requires_aligned_response_arrays():
    with pytest.raises(ValueError, match="do not align"):
        Trajectory(
            trajectory_id="trajectory-1",
            session_id="session-1",
            chain_id="chain-1",
            segment_id="segment-1",
            parent_chain_id=None,
            split_reason=None,
            prompt_ids=(1, 2),
            response_ids=(3, 4),
            generation_mask=(1,),
            loss_mask=None,
            loss_weight=None,
            response_logprobs=(-0.1, -0.2),
            token_provenance=(
                TokenProvenance("turn-1", "generated"),
                TokenProvenance("turn-1", "generated"),
            ),
            routed_experts=None,
        )


def test_trajectory_bundle_rejects_duplicate_ids():
    trajectory = Trajectory(
        trajectory_id="trajectory-1",
        session_id="session-1",
        chain_id="chain-1",
        segment_id="segment-1",
        parent_chain_id=None,
        split_reason=None,
        prompt_ids=(1, 2),
        response_ids=(3,),
        generation_mask=(1,),
        loss_mask=(1,),
        loss_weight=None,
        response_logprobs=(-0.1,),
        token_provenance=(TokenProvenance("turn-1", "generated"),),
        routed_experts=None,
    )
    with pytest.raises(ValueError, match="unique"):
        TrajectoryBundle(
            session_id="session-1",
            trajectories=(trajectory, trajectory),
            lineage={},
            metadata={},
        )


def test_proxy_error_wire_shape_does_not_include_missing_ids():
    error = ProxyError("boom", details={"safe": True})

    assert error.to_dict() == {
        "error": {
            "code": "proxy_error",
            "message": "boom",
            "retryable": False,
            "state_mutated": False,
            "details": {"safe": True},
        }
    }
