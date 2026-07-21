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

import importlib
import logging
import uuid
from collections.abc import Mapping
from typing import Any

import ray
from omegaconf import OmegaConf

from ..client_sdk.executor import AgentExecutor
from ..client_sdk.transport import RayTransportClient
from .config import AgentServiceStartupConfig, InferenceSpec
from .rollout_adapter import RolloutAdapter

logger = logging.getLogger(__name__)


def _load_class(fqn: str) -> type:
    if "." not in fqn:
        raise ValueError(f"Invalid class {fqn!r}; expected a fully qualified class name")
    module_name, class_name = fqn.rsplit(".", 1)
    loaded_class = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(loaded_class, type) and not hasattr(loaded_class, "remote"):
        raise TypeError(f"Configured class {fqn!r} is not a class or Ray ActorClass")
    return loaded_class


def _resolve_service_config(config: Any) -> Mapping[str, Any]:
    service_config = OmegaConf.to_container(config.agent_service, resolve=True)
    if not isinstance(service_config, Mapping):
        raise TypeError("config.agent_service must resolve to a mapping")
    return service_config


def _build_proxy_config(config: Any, service_config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the wire-safe Proxy config owned by the Agent Service.

    Keep the original model artifact locations rather than TaskRunner-local
    copies. The Proxy may run on another node (or behind HTTP in a later
    version), so it must resolve the tokenizer artifact in its own runtime.
    """

    configured_proxy = service_config.get("proxy")
    if configured_proxy is not None and not isinstance(configured_proxy, Mapping):
        raise TypeError("config.agent_service.proxy must resolve to a mapping or null")

    model_config = OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True)
    if not isinstance(model_config, Mapping):
        raise TypeError("config.actor_rollout_ref.model must resolve to a mapping")

    proxy_config = dict(configured_proxy or {})
    proxy_config["model_config"] = dict(model_config)
    return proxy_config


def _build_startup_config(trainer: Any, config: Any, service_config: Mapping[str, Any]) -> AgentServiceStartupConfig:
    # The replica list is discovered once after init_workers() and stays fixed
    # for the lifetime of this Agent Service instance.
    return AgentServiceStartupConfig(
        execution_backend=service_config["execution_backend"],
        inference=InferenceSpec(
            replica_endpoints=trainer.get_rollout_inference_addresses(),
            upstream_protocol=service_config["upstream_protocol"],
        ),
        proxy=_build_proxy_config(config, service_config),
        admission=service_config.get("admission"),
        default_lifecycle=service_config.get("default_lifecycle"),
    )


class VerlAgentServiceRuntime:
    """TaskRunner-owned Agent Service lifecycle for verl training."""

    def __init__(
        self,
        *,
        trainer: Any,
        config: Any,
    ) -> None:
        self._trainer = trainer
        self._config = config
        self._service_config = _resolve_service_config(config)
        self._ray_actor: Any = None
        self._executor: AgentExecutor | None = None
        self._rollout_adapter: Any = None
        self._shutdown_timeout_seconds = float(self._service_config["shutdown_timeout_seconds"])
        self._started = False
        self._closed = False

    def start(self) -> None:
        """Start Agent Service and initialize its Driver-side executor."""

        if self._closed:
            raise RuntimeError("VerlAgentServiceRuntime is closed")
        if self._started:
            raise RuntimeError("VerlAgentServiceRuntime is already started")

        try:
            startup_config = _build_startup_config(self._trainer, self._config, self._service_config)
            ray_actor_config = self._service_config["ray_actor"]
            service_endpoint = self._create_ray_actor(
                startup_config=startup_config,
                actor_config=ray_actor_config,
            )
            ready_timeout_seconds = self._service_config["ready_timeout_seconds"]
            ray.get(
                self._ray_actor.wait_ready.remote(ready_timeout_seconds),
                timeout=ready_timeout_seconds,
            )
            transport_client = RayTransportClient(
                service_endpoint,
                rpc_timeout_seconds=float(self._service_config["rpc_timeout_seconds"]),
            )
            self._executor = AgentExecutor(
                transport_client,
                max_in_flight=self._service_config.get("max_in_flight"),
            )
            self._started = True
        except BaseException:
            self.close(raise_on_error=False)
            raise

    def get_rollout_adapter(
        self,
        *,
        tokenizer: Any,
        processor: Any = None,
    ) -> Any:
        """Create once and return the rollout adapter for the running service."""

        if self._closed:
            raise RuntimeError("VerlAgentServiceRuntime is closed")
        if not self._started or self._executor is None:
            raise RuntimeError("VerlAgentServiceRuntime must be started before getting its rollout adapter")
        if self._rollout_adapter is None:
            try:
                self._rollout_adapter = RolloutAdapter(
                    executor=self._executor,
                    config=self._config,
                    tokenizer=tokenizer,
                    processor=processor,
                )
            except BaseException:
                self.close(raise_on_error=False)
                raise
        return self._rollout_adapter

    def _create_ray_actor(
        self,
        *,
        startup_config: AgentServiceStartupConfig,
        actor_config: Mapping[str, Any],
    ) -> dict[str, str]:
        options = dict(actor_config.get("actor_options") or {})
        actor_name = options.setdefault("name", f"verl-agent-service-{uuid.uuid4().hex}")
        if not isinstance(actor_name, str) or not actor_name:
            raise ValueError("agent_service.ray_actor.actor_options.name must be a non-empty string")

        configured_class = actor_config["actor_class"]
        actor_class = _load_class(configured_class) if isinstance(configured_class, str) else configured_class
        ray_actor_class = actor_class if hasattr(actor_class, "remote") else ray.remote(actor_class)
        namespace = getattr(ray.get_runtime_context(), "namespace", None) or None
        endpoint = {"transport": "ray", "actor_name": actor_name}
        if namespace is not None:
            endpoint["namespace"] = namespace
        self._ray_actor = ray_actor_class.options(**options).remote(startup_config.to_dict())
        return endpoint

    def close(self, *, raise_on_error: bool = True) -> None:
        if self._closed:
            return
        self._closed = True

        cleanups: list[tuple[str, Any]] = []
        if self._rollout_adapter is not None:
            cleanups.append(("cancel in-flight Agent Service Tasks", self._rollout_adapter.cancel_in_flight))
        if self._executor is not None:
            cleanups.append(("close Agent Service executor", self._executor.close))
        if self._ray_actor is not None:
            cleanups.append(("stop Agent Service Ray actor", self._stop_ray_actor))

        errors: list[BaseException] = []
        for description, cleanup in cleanups:
            try:
                cleanup()
            except BaseException as exc:
                errors.append(exc)
                logger.exception("Failed to %s", description)
        if errors and raise_on_error:
            raise errors[0]

    def _stop_ray_actor(self) -> None:
        try:
            ray.get(self._ray_actor.stop.remote(self._shutdown_timeout_seconds))
        finally:
            ray.kill(self._ray_actor, no_restart=True)
