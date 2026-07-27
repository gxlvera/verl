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

"""Agent Service Proxy contracts and bootstrap helpers."""

from .base import Proxy
from .canonical import Canonicalizer, common_prefix_length, exact_prefix_length, fingerprint_tools, message_prefixes
from .codec import AssistantResponseDecoder, TextResponseDecoder, VerlToolResponseDecoder
from .continuous_tokens import (
    ContinuousTokenCodec,
    ContinuousTokenIntegrationError,
    ContinuousTokenMerge,
    create_continuous_token_codec,
)
from .errors import (
    ContinuousTokenMergeError,
    InvalidSessionTokenError,
    MalformedRequestError,
    ProxyError,
    SessionBudgetExceededError,
    SessionConflictError,
    SessionInactiveError,
    TokenStateCorruptionError,
    UnknownSessionError,
    UnsupportedRequestError,
    UpstreamGenerationError,
    UpstreamTimeoutError,
)
from .factory import build_hosted_proxy
from .materializer import LossMaterializationPolicy, TrajectoryMaterializer
from .models import (
    AgentSessionHandle,
    BoundaryAdjustment,
    CanonicalMessage,
    CanonicalPrompt,
    CanonicalTool,
    ChainLifecycleState,
    ChainState,
    DeliveryStatus,
    FrontendProtocol,
    GenerationSpec,
    InternalGenerationRequest,
    MessagePrefix,
    SessionEvent,
    SessionLifecycleState,
    SplitReason,
    TokenProvenance,
    TokenSegmentState,
    Trajectory,
    TrajectoryBundle,
    TurnRecord,
)
from .processor import ProcessedGeneration, ProxyRequestProcessor
from .routing import ChainRouter, RoutingAction, RoutingDecision
from .server import ProxyHttpServer, create_proxy_app
from .service import HostedProxy, InMemoryProxy
from .session import PreparedTurn, ProxySession, ProxySessionManager
from .tokenizer import load_tokenizer_and_processor_from_model_config, load_tokenizer_from_model_config
from .upstream import (
    CallableTokenGenerationClient,
    HttpTokenGenerationClient,
    TokenGenerationOutput,
    TokenGenerationRequest,
    UpstreamGenerator,
)

__all__ = [
    "AgentSessionHandle",
    "AssistantResponseDecoder",
    "BoundaryAdjustment",
    "CanonicalMessage",
    "CanonicalPrompt",
    "CanonicalTool",
    "Canonicalizer",
    "ChainLifecycleState",
    "ChainRouter",
    "ChainState",
    "CallableTokenGenerationClient",
    "ContinuousTokenCodec",
    "ContinuousTokenIntegrationError",
    "ContinuousTokenMergeError",
    "ContinuousTokenMerge",
    "DeliveryStatus",
    "FrontendProtocol",
    "GenerationSpec",
    "InternalGenerationRequest",
    "InMemoryProxy",
    "InvalidSessionTokenError",
    "MalformedRequestError",
    "LossMaterializationPolicy",
    "MessagePrefix",
    "HttpTokenGenerationClient",
    "HostedProxy",
    "Proxy",
    "ProxyError",
    "ProxyHttpServer",
    "PreparedTurn",
    "ProcessedGeneration",
    "ProxySession",
    "ProxySessionManager",
    "ProxyRequestProcessor",
    "RoutingAction",
    "RoutingDecision",
    "SessionBudgetExceededError",
    "SessionConflictError",
    "SessionEvent",
    "SessionInactiveError",
    "SessionLifecycleState",
    "SplitReason",
    "TokenGenerationOutput",
    "TokenGenerationRequest",
    "TokenProvenance",
    "TokenSegmentState",
    "TokenStateCorruptionError",
    "TextResponseDecoder",
    "Trajectory",
    "TrajectoryBundle",
    "TrajectoryMaterializer",
    "TurnRecord",
    "UnknownSessionError",
    "UpstreamGenerator",
    "UnsupportedRequestError",
    "UpstreamGenerationError",
    "UpstreamTimeoutError",
    "VerlToolResponseDecoder",
    "create_continuous_token_codec",
    "create_proxy_app",
    "build_hosted_proxy",
    "common_prefix_length",
    "exact_prefix_length",
    "fingerprint_tools",
    "load_tokenizer_and_processor_from_model_config",
    "load_tokenizer_from_model_config",
    "message_prefixes",
]
