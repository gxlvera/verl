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

"""Build the concrete hosted Proxy from AgentService startup configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .canonical import Canonicalizer
from .codec import TextResponseDecoder, VerlToolResponseDecoder
from .continuous_tokens import create_continuous_token_codec
from .materializer import TrajectoryMaterializer
from .processor import ProxyRequestProcessor
from .server import ProxyHttpServer, create_proxy_app
from .service import HostedProxy
from .session import ProxySessionManager
from .tokenizer import load_tokenizer_from_model_config
from .upstream import HttpTokenGenerationClient, UpstreamGenerator


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return dict(value)


def build_hosted_proxy(
    startup_config: Mapping[str, Any] | Any,
    *,
    source_resolver: Callable[[str], str] | None = None,
    upstream: UpstreamGenerator | None = None,
) -> HostedProxy:
    config = _mapping(startup_config, "startup_config")
    inference = _mapping(config.get("inference"), "startup_config.inference")
    proxy_config = _mapping(config.get("proxy") or {}, "startup_config.proxy")
    upstream_protocol = inference.get("upstream_protocol")
    if upstream_protocol != "generate":
        raise ValueError(
            "Agent Service Proxy requires inference.upstream_protocol='generate' to preserve exact token IDs"
        )
    endpoints = inference.get("replica_endpoints")
    if not isinstance(endpoints, list | tuple):
        raise TypeError("startup_config.inference.replica_endpoints must be a sequence")

    model_config = _mapping(proxy_config.get("model_config"), "startup_config.proxy.model_config")
    tokenizer = load_tokenizer_from_model_config(
        model_config,
        source_resolver=source_resolver,
    )
    continuous_config = _mapping(
        proxy_config.get("continuous_token") or {},
        "startup_config.proxy.continuous_token",
    )
    codec = create_continuous_token_codec(
        tokenizer,
        model_family=continuous_config.get("model_family", "auto"),
        model_path=model_config.get("path"),
        tokenizer_name_or_path=model_config.get("tokenizer_path"),
        chat_template_kwargs=continuous_config.get("chat_template_kwargs"),
    )
    tool_parser = proxy_config.get("tool_parser")
    if tool_parser is None:
        response_decoder = TextResponseDecoder(tokenizer)
    else:
        if not isinstance(tool_parser, str) or not tool_parser:
            raise ValueError("startup_config.proxy.tool_parser must be a non-empty string or null")
        response_decoder = VerlToolResponseDecoder(tokenizer, tool_parser_name=tool_parser)

    if upstream is None:
        upstream_client: UpstreamGenerator = HttpTokenGenerationClient(
            generate_path=proxy_config.get("upstream_generate_path", "/agent_service/generate"),
            timeout_seconds=float(proxy_config.get("upstream_timeout_seconds", 120)),
        )
    else:
        upstream_client = upstream

    host = proxy_config.get("host", "127.0.0.1")
    port = proxy_config.get("port", 0)
    advertised_host = proxy_config.get("advertised_host")
    manager = ProxySessionManager(
        replica_endpoints=endpoints,
        frontend_base_url="http://pending",
        continuous_token_codec=codec,
        canonicalizer=Canonicalizer(ignored_message_fields=proxy_config.get("ignored_message_fields", ())),
        model_name=proxy_config.get("model_name") or model_config.get("path"),
    )
    processor = ProxyRequestProcessor(
        upstream=upstream_client,
        response_decoder=response_decoder,
    )
    app = create_proxy_app(session_manager=manager, processor=processor)
    http_server = ProxyHttpServer(
        app,
        host=host,
        port=port,
        advertised_host=advertised_host,
        ready_timeout_seconds=float(proxy_config.get("ready_timeout_seconds", 30)),
        log_level=proxy_config.get("log_level", "warning"),
    )
    materializer = TrajectoryMaterializer(loss_policy=proxy_config.get("loss_materialization_policy", "per_leaf"))
    return HostedProxy(
        session_manager=manager,
        http_server=http_server,
        materializer=materializer,
        finalize_timeout_seconds=float(proxy_config.get("finalize_timeout_seconds", 60)),
        upstream_client=upstream_client if upstream is None else None,
    )
