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

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_service.proxy.models import TrajectoryBundle

from .base import TrajectorySelectionContext, TrajectorySelectionError


def _require_empty_config(config: Mapping[str, Any], selector_name: str) -> None:
    if config:
        raise TrajectorySelectionError(f"{selector_name} selector does not accept configuration")


class AllTrajectorySelector:
    name = "all"
    version = "v0"

    def select(
        self,
        bundle: TrajectoryBundle,
        context: TrajectorySelectionContext,
        config: Mapping[str, Any],
    ) -> list[str]:
        del context
        _require_empty_config(config, self.name)
        return [trajectory.trajectory_id for trajectory in bundle.trajectories]


class LongestTrajectorySelector:
    name = "longest"
    version = "v0"

    def select(
        self,
        bundle: TrajectoryBundle,
        context: TrajectorySelectionContext,
        config: Mapping[str, Any],
    ) -> list[str]:
        del context
        _require_empty_config(config, self.name)
        if not bundle.trajectories:
            return []

        def rank(trajectory):
            generated_tokens = sum(trajectory.generation_mask)
            response_tokens = len(trajectory.response_ids)
            turn_count = int(trajectory.metadata.get("turn_count", 0))
            materialization_order = int(trajectory.metadata.get("materialization_order", 0))
            return generated_tokens, response_tokens, turn_count, -materialization_order

        return [max(bundle.trajectories, key=rank).trajectory_id]
