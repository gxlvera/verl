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

"""Control-plane protocol implemented by the experiment-scoped Proxy."""

from __future__ import annotations

from typing import Protocol

from .models import AgentSessionHandle, GenerationSpec, TrajectoryBundle


class Proxy(Protocol):
    async def start(self) -> None: ...

    async def create_session(
        self,
        *,
        session_id: str,
        generation_spec: GenerationSpec,
    ) -> AgentSessionHandle: ...

    async def finalize_session(self, session_id: str) -> TrajectoryBundle: ...

    async def abort_session(self, session_id: str) -> None: ...

    async def close(self) -> None: ...
