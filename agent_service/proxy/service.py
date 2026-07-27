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

"""Concrete in-memory control plane for the V0 Proxy."""

from __future__ import annotations

import inspect
from typing import Any

from .errors import SessionConflictError
from .materializer import TrajectoryMaterializer
from .models import AgentSessionHandle, GenerationSpec, TrajectoryBundle
from .server import ProxyHttpServer
from .session import ProxySessionManager


class InMemoryProxy:
    def __init__(
        self,
        *,
        session_manager: ProxySessionManager,
        materializer: TrajectoryMaterializer | None = None,
        finalize_timeout_seconds: float = 60.0,
    ) -> None:
        if finalize_timeout_seconds <= 0:
            raise ValueError("finalize_timeout_seconds must be greater than zero")
        self.session_manager = session_manager
        self.materializer = materializer or TrajectoryMaterializer()
        self.finalize_timeout_seconds = finalize_timeout_seconds
        self._started = False
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Proxy is closed")
        self._started = True

    def _require_started(self) -> None:
        if self._closed:
            raise RuntimeError("Proxy is closed")
        if not self._started:
            raise RuntimeError("Proxy must be started first")

    async def create_session(
        self,
        *,
        session_id: str,
        generation_spec: GenerationSpec,
    ) -> AgentSessionHandle:
        self._require_started()
        return await self.session_manager.create_session(
            session_id=session_id,
            generation_spec=generation_spec,
        )

    async def finalize_session(self, session_id: str) -> TrajectoryBundle:
        self._require_started()
        session = await self.session_manager.get_session(session_id)
        cached = await session.cached_finalized_bundle()
        if cached is not None:
            return cached
        await session.begin_finalization(timeout_seconds=self.finalize_timeout_seconds)
        cached = await session.cached_finalized_bundle()
        if cached is not None:
            return cached
        bundle = self.materializer.materialize(session)
        try:
            return await session.complete_finalization(bundle)
        except SessionConflictError:
            cached = await session.cached_finalized_bundle()
            if cached is not None:
                return cached
            raise

    async def abort_session(self, session_id: str) -> None:
        self._require_started()
        await self.session_manager.abort_session(session_id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.session_manager.close()


class HostedProxy(InMemoryProxy):
    """In-memory control plane plus a real experiment-scoped HTTP listener."""

    def __init__(
        self,
        *,
        session_manager: ProxySessionManager,
        http_server: ProxyHttpServer,
        materializer: TrajectoryMaterializer | None = None,
        finalize_timeout_seconds: float = 60.0,
        upstream_client: Any | None = None,
    ) -> None:
        super().__init__(
            session_manager=session_manager,
            materializer=materializer,
            finalize_timeout_seconds=finalize_timeout_seconds,
        )
        self.http_server = http_server
        self.upstream_client = upstream_client

    async def start(self) -> None:
        endpoint = await self.http_server.start()
        self.session_manager.frontend_base_url = endpoint
        try:
            await super().start()
        except BaseException:
            await self.http_server.close()
            raise

    async def close(self) -> None:
        if self._closed:
            return
        await self.http_server.close()
        await super().close()
        close = getattr(self.upstream_client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
