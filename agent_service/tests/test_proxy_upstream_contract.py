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

from agent_service.proxy.upstream import TokenGenerationOutput, TokenGenerationRequest


def test_token_generation_request_copies_mutable_input():
    sampling_params = {"temperature": 0.7}
    request = TokenGenerationRequest(
        request_id="request-1",
        session_id="session-1",
        prompt_token_ids=(1, 2, 3),
        sampling_params=sampling_params,
    )
    sampling_params["temperature"] = 1.0

    assert request.prompt_token_ids == (1, 2, 3)
    assert request.sampling_params == {"temperature": 0.7}


def test_token_generation_output_requires_aligned_logprobs():
    with pytest.raises(ValueError, match="logprobs length"):
        TokenGenerationOutput(
            token_ids=(4, 5),
            logprobs=(-0.1,),
            finish_reason="stop",
        )


@pytest.mark.parametrize("token_ids", [(1, -1), (1, True)])
def test_token_generation_contract_rejects_invalid_token_ids(token_ids):
    with pytest.raises(ValueError, match="non-negative integer"):
        TokenGenerationRequest(
            request_id="request-1",
            session_id="session-1",
            prompt_token_ids=token_ids,
            sampling_params={},
        )
