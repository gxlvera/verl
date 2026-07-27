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

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_service.proxy.models import TrajectoryBundle


def _json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain JSON-compatible values") from exc


class TrajectorySelectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrajectorySelectionSpec:
    strategy: str = "longest"
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, str) or not self.strategy:
            raise ValueError("trajectory selection strategy must be a non-empty string")
        object.__setattr__(self, "config", _json_mapping(self.config, "trajectory selection config"))

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "config": dict(self.config)}


@dataclass(frozen=True)
class TrajectorySelectionContext:
    session_metadata: Mapping[str, Any] = field(default_factory=dict)
    task_metadata: Mapping[str, Any] = field(default_factory=dict)
    rewards: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("session_metadata", "task_metadata", "rewards"):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), name))


class TrajectorySelector(Protocol):
    name: str
    version: str

    def select(
        self,
        bundle: TrajectoryBundle,
        context: TrajectorySelectionContext,
        config: Mapping[str, Any],
    ) -> list[str]: ...


@dataclass(frozen=True)
class TrajectorySelectionResult:
    selector_name: str
    selector_version: str
    config_fingerprint: str
    candidate_trajectory_ids: tuple[str, ...]
    selected_trajectory_ids: tuple[str, ...]
    selected_bundle: TrajectoryBundle
    diagnostics: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector_name": self.selector_name,
            "selector_version": self.selector_version,
            "config_fingerprint": self.config_fingerprint,
            "candidate_trajectory_ids": list(self.candidate_trajectory_ids),
            "selected_trajectory_ids": list(self.selected_trajectory_ids),
            "selected_bundle": self.selected_bundle.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }
