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

"""AgentService-side finalization: retain complete bundle, then select a view."""

from __future__ import annotations

from agent_service.proxy.base import Proxy
from agent_service.proxy.models import TrajectoryBundle

from .base import (
    TrajectorySelectionContext,
    TrajectorySelectionResult,
    TrajectorySelectionSpec,
)
from .registry import TrajectorySelectorRegistry


class AgentServiceTrajectoryFinalizer:
    def __init__(self, registry: TrajectorySelectorRegistry | None = None) -> None:
        self.registry = registry or TrajectorySelectorRegistry()
        self._complete_bundles: dict[str, TrajectoryBundle] = {}
        self._selection_results: dict[str, TrajectorySelectionResult] = {}

    async def finalize(
        self,
        *,
        proxy: Proxy,
        session_id: str,
        selection_spec: TrajectorySelectionSpec | None = None,
        context: TrajectorySelectionContext | None = None,
    ) -> TrajectorySelectionResult:
        existing = self._selection_results.get(session_id)
        if existing is not None:
            return existing
        bundle = self._complete_bundles.get(session_id)
        if bundle is None:
            bundle = await proxy.finalize_session(session_id)
            self._complete_bundles[session_id] = bundle
        result = self.registry.select(
            bundle,
            spec=selection_spec,
            context=context,
        )
        self._selection_results[session_id] = result
        return result

    def get_complete_bundle(self, session_id: str) -> TrajectoryBundle | None:
        return self._complete_bundles.get(session_id)
