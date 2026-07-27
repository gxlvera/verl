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

import hashlib
import json

from agent_service.proxy.models import TrajectoryBundle

from .base import (
    TrajectorySelectionContext,
    TrajectorySelectionError,
    TrajectorySelectionResult,
    TrajectorySelectionSpec,
    TrajectorySelector,
)
from .builtins import AllTrajectorySelector, LongestTrajectorySelector


class TrajectorySelectorRegistry:
    def __init__(self) -> None:
        self._selectors: dict[str, TrajectorySelector] = {}
        self.register(AllTrajectorySelector())
        self.register(LongestTrajectorySelector())

    def register(self, selector: TrajectorySelector, *, replace: bool = False) -> None:
        name = getattr(selector, "name", None)
        version = getattr(selector, "version", None)
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            raise ValueError("selector name and version must be non-empty strings")
        if name in self._selectors and not replace:
            raise ValueError(f"Trajectory selector {name!r} is already registered")
        self._selectors[name] = selector

    def resolve(self, strategy: str) -> TrajectorySelector:
        try:
            return self._selectors[strategy]
        except KeyError as exc:
            raise TrajectorySelectionError(f"Unknown server-registered trajectory selector: {strategy!r}") from exc

    def select(
        self,
        bundle: TrajectoryBundle,
        *,
        spec: TrajectorySelectionSpec | None = None,
        context: TrajectorySelectionContext | None = None,
    ) -> TrajectorySelectionResult:
        spec = spec or TrajectorySelectionSpec()
        context = context or TrajectorySelectionContext()
        selector = self.resolve(spec.strategy)
        selected_ids = selector.select(bundle, context, spec.config)
        if not isinstance(selected_ids, list) or any(not isinstance(item, str) for item in selected_ids):
            raise TrajectorySelectionError("Trajectory selector must return list[str]")
        if bundle.trajectories and not selected_ids:
            raise TrajectorySelectionError("Trajectory selector returned an empty selection for a non-empty bundle")
        if len(set(selected_ids)) != len(selected_ids):
            raise TrajectorySelectionError("Trajectory selector returned duplicate trajectory IDs")

        candidates = {trajectory.trajectory_id: trajectory for trajectory in bundle.trajectories}
        unknown = [trajectory_id for trajectory_id in selected_ids if trajectory_id not in candidates]
        if unknown:
            raise TrajectorySelectionError(f"Trajectory selector returned unknown IDs: {unknown}")
        selected_bundle = TrajectoryBundle(
            session_id=bundle.session_id,
            trajectories=tuple(candidates[trajectory_id] for trajectory_id in selected_ids),
            lineage=bundle.lineage,
            metadata={
                **bundle.metadata,
                "selection": {
                    "selector_name": selector.name,
                    "selector_version": selector.version,
                    "selected_trajectory_ids": selected_ids,
                },
            },
        )
        config_json = json.dumps(
            spec.config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return TrajectorySelectionResult(
            selector_name=selector.name,
            selector_version=selector.version,
            config_fingerprint=hashlib.sha256(config_json.encode()).hexdigest(),
            candidate_trajectory_ids=tuple(candidates),
            selected_trajectory_ids=tuple(selected_ids),
            selected_bundle=selected_bundle,
            diagnostics={
                "candidate_count": len(candidates),
                "selected_count": len(selected_ids),
            },
        )
