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

"""Task-scoped ProxySession lifecycle, chain reservations, and atomic commit."""

from __future__ import annotations

import asyncio
import secrets
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from .canonical import Canonicalizer
from .continuous_tokens import ContinuousTokenCodec, ContinuousTokenIntegrationError
from .errors import (
    ContinuousTokenMergeError,
    InvalidSessionTokenError,
    SessionBudgetExceededError,
    SessionConflictError,
    SessionInactiveError,
    TokenStateCorruptionError,
    UnknownSessionError,
)
from .models import (
    AgentSessionHandle,
    BoundaryAdjustment,
    CanonicalMessage,
    CanonicalPrompt,
    ChainLifecycleState,
    ChainState,
    DeliveryStatus,
    GenerationSpec,
    InternalGenerationRequest,
    SessionEvent,
    SessionLifecycleState,
    TokenSegmentState,
    TrajectoryBundle,
    TurnRecord,
)
from .routing import ChainRouter, RoutingAction
from .upstream import TokenGenerationOutput


@dataclass(frozen=True)
class PreparedTurn:
    request_id: str
    turn_id: str
    chain_id: str
    segment_id: str
    prompt: CanonicalPrompt
    request: InternalGenerationRequest
    context_ids: tuple[int, ...]
    exact_prefix_ids: tuple[int, ...]
    appended_context_ids: tuple[int, ...]
    removed_prefix_ids: tuple[int, ...]
    sampling_params: Mapping[str, Any]
    is_new_chain: bool
    estimated_token_budget: int


class ProxySession:
    """One task-scoped collection of every active and completed chain."""

    def __init__(
        self,
        *,
        session_id: str,
        auth_token: str,
        generation_spec: GenerationSpec,
        upstream_replica: str,
        model_name: str | None,
        continuous_token_codec: ContinuousTokenCodec,
        canonicalizer: Canonicalizer,
        router: ChainRouter | None = None,
    ) -> None:
        self.session_id = session_id
        self.auth_token = auth_token
        self.generation_spec = generation_spec
        self.upstream_replica = upstream_replica
        self.model_name = model_name
        self.continuous_token_codec = continuous_token_codec
        self.canonicalizer = canonicalizer
        self.router = router or ChainRouter()

        self.state = SessionLifecycleState.ACTIVE
        self.chains: OrderedDict[str, ChainState] = OrderedDict()
        self.events: list[SessionEvent] = []
        self.model_request_count = 0
        self.total_token_count = 0
        self._reserved_token_count = 0
        self._in_flight: dict[str, PreparedTurn] = {}
        self._sequence = 0
        self._finalized_bundle: TrajectoryBundle | None = None
        self._condition = asyncio.Condition()

    @property
    def in_flight_request_ids(self) -> tuple[str, ...]:
        return tuple(self._in_flight)

    def _next_sequence_locked(self) -> int:
        self._sequence += 1
        return self._sequence

    def _record_event_locked(
        self,
        event_type: str,
        *,
        request_id: str | None = None,
        chain_id: str | None = None,
        failure_reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.events.append(
            SessionEvent(
                sequence=self._next_sequence_locked(),
                event_type=event_type,
                request_id=request_id,
                chain_id=chain_id,
                failure_reason=failure_reason,
                metadata=metadata or {},
            )
        )

    def _require_active_locked(self, request_id: str | None = None) -> None:
        if self.state is not SessionLifecycleState.ACTIVE:
            raise SessionInactiveError(
                f"Session {self.session_id} is {self.state.value}",
                session_id=self.session_id,
                request_id=request_id,
                details={"state": self.state.value},
            )

    async def require_active(self, request_id: str | None = None) -> None:
        async with self._condition:
            self._require_active_locked(request_id)

    def _merged_sampling_params(self, request: InternalGenerationRequest) -> dict[str, Any]:
        sampling_params = dict(self.generation_spec.sampling_params)
        sampling_params.update(request.sampling_params)
        requested_max = sampling_params.pop(
            "max_tokens",
            sampling_params.pop("max_new_tokens", self.generation_spec.max_new_tokens),
        )
        if not isinstance(requested_max, int) or isinstance(requested_max, bool) or requested_max <= 0:
            raise SessionBudgetExceededError(
                "max_tokens must be a positive integer",
                session_id=self.session_id,
            )
        sampling_params["max_tokens"] = min(requested_max, self.generation_spec.max_new_tokens)
        sampling_params["logprobs"] = True
        return sampling_params

    def _check_budget_locked(self, context_length: int, max_new_tokens: int, request_id: str) -> int:
        if self.model_request_count >= self.generation_spec.max_model_requests:
            raise SessionBudgetExceededError(
                "Session model-request budget exceeded",
                session_id=self.session_id,
                request_id=request_id,
                details={
                    "max_model_requests": self.generation_spec.max_model_requests,
                    "model_request_count": self.model_request_count,
                },
            )
        estimate = context_length + max_new_tokens
        projected = self.total_token_count + self._reserved_token_count + estimate
        if projected > self.generation_spec.max_total_tokens:
            raise SessionBudgetExceededError(
                "Session total-token budget exceeded",
                session_id=self.session_id,
                request_id=request_id,
                details={
                    "max_total_tokens": self.generation_spec.max_total_tokens,
                    "projected_total_tokens": projected,
                },
            )
        return estimate

    async def prepare_turn(
        self,
        *,
        request_id: str,
        request: InternalGenerationRequest,
        prompt: CanonicalPrompt,
    ) -> PreparedTurn:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        async with self._condition:
            self._require_active_locked(request_id)
            if request_id in self._in_flight:
                raise SessionConflictError(
                    f"Request {request_id} is already in flight",
                    session_id=self.session_id,
                    request_id=request_id,
                )

            decision = self.router.select(
                prompt,
                self.chains.values(),
                context_compaction=request.request_metadata.get("context_compaction") is True,
            )
            canonical_messages = [message.to_dict() for message in prompt.messages]
            canonical_tools = [tool.to_dict() for tool in prompt.tools]
            selected_chain: ChainState | None = None
            try:
                if decision.action is RoutingAction.REUSE:
                    selected_chain = self.chains[decision.chain_id]
                    exact_prefix_ids = selected_chain.current_segment.exact_token_ids
                    merge = self.continuous_token_codec.merge_non_assistant(
                        exact_prefix_ids=exact_prefix_ids,
                        previous_messages=[message.to_dict() for message in selected_chain.message_history],
                        updated_messages=canonical_messages,
                        tools=canonical_tools,
                    )
                    context_ids = merge.context_ids
                    appended_context_ids = merge.appended_context_ids
                    removed_prefix_ids = merge.removed_prefix_ids
                    if len(removed_prefix_ids) > len(selected_chain.current_segment.response_ids):
                        raise ContinuousTokenIntegrationError(
                            "Continuous Token attempted to remove tokens from the immutable initial prompt",
                            diagnostics={
                                **merge.diagnostics,
                                "segment_response_length": len(selected_chain.current_segment.response_ids),
                            },
                        )
                    is_new_chain = False
                else:
                    merge = self.continuous_token_codec.encode_initial(
                        canonical_messages,
                        tools=canonical_tools,
                    )
                    context_ids = merge.context_ids
                    exact_prefix_ids = context_ids
                    appended_context_ids = ()
                    removed_prefix_ids = ()
                    is_new_chain = True
            except ContinuousTokenIntegrationError as exc:
                self._record_event_locked(
                    "request_failed",
                    request_id=request_id,
                    chain_id=selected_chain.chain_id if selected_chain is not None else None,
                    failure_reason=exc.failure_reason,
                    metadata=exc.diagnostics,
                )
                raise ContinuousTokenMergeError(
                    str(exc),
                    session_id=self.session_id,
                    request_id=request_id,
                    state_mutated=False,
                    details=exc.diagnostics,
                ) from exc

            sampling_params = self._merged_sampling_params(request)
            estimated_token_budget = self._check_budget_locked(
                len(context_ids),
                sampling_params["max_tokens"],
                request_id,
            )
            turn_id = f"turn-{uuid4().hex}"
            if is_new_chain:
                chain_id = f"chain-{uuid4().hex}"
                segment_id = f"segment-{uuid4().hex}"
                selected_chain = ChainState(
                    chain_id=chain_id,
                    parent_chain_id=decision.parent_chain_id,
                    branch_message_index=decision.branch_message_index,
                    split_reason=decision.split_reason,
                    message_history=list(prompt.messages),
                    message_prefix_hashes=[
                        prefix.rolling_hash for prefix in self.canonicalizer.prefix_hashes(prompt.messages)
                    ],
                    tools=prompt.tools,
                    tools_fingerprint=prompt.tools_fingerprint,
                    segments=[TokenSegmentState(segment_id=segment_id, prompt_ids=list(context_ids))],
                    state=ChainLifecycleState.RESERVED,
                    updated_sequence=self._next_sequence_locked(),
                    materialization_order=len(self.chains),
                    reserved_by=request_id,
                    pending_request_messages=prompt.messages,
                )
                self.chains[chain_id] = selected_chain
            else:
                assert selected_chain is not None
                chain_id = selected_chain.chain_id
                segment_id = selected_chain.current_segment.segment_id
                selected_chain.state = ChainLifecycleState.RESERVED
                selected_chain.reserved_by = request_id
                selected_chain.pending_request_messages = prompt.messages

            prepared = PreparedTurn(
                request_id=request_id,
                turn_id=turn_id,
                chain_id=chain_id,
                segment_id=segment_id,
                prompt=prompt,
                request=request,
                context_ids=context_ids,
                exact_prefix_ids=exact_prefix_ids,
                appended_context_ids=appended_context_ids,
                removed_prefix_ids=removed_prefix_ids,
                sampling_params=sampling_params,
                is_new_chain=is_new_chain,
                estimated_token_budget=estimated_token_budget,
            )
            self._in_flight[request_id] = prepared
            self.model_request_count += 1
            self._reserved_token_count += estimated_token_budget
            self._record_event_locked(
                "request_prepared",
                request_id=request_id,
                chain_id=chain_id,
                metadata={
                    "routing_action": decision.action.value,
                    "split_reason": decision.split_reason.value if decision.split_reason else None,
                    "context_token_count": len(context_ids),
                },
            )
            return prepared

    async def commit_turn(
        self,
        prepared: PreparedTurn,
        *,
        output: TokenGenerationOutput,
        response_message: CanonicalMessage,
        delivery_status: DeliveryStatus = DeliveryStatus.UNKNOWN,
        metadata: Mapping[str, Any] | None = None,
    ) -> TurnRecord:
        response_message = self.canonicalizer.canonicalize_message(response_message.to_dict())
        # Assistant tokens are token truth returned by the backend. Appending
        # them is deliberately a direct concatenation operation; no tokenizer
        # or decoded response text participates in this state transition.
        async with self._condition:
            registered = self._in_flight.get(prepared.request_id)
            if registered != prepared:
                raise SessionConflictError(
                    "Prepared request reservation is missing or stale",
                    session_id=self.session_id,
                    request_id=prepared.request_id,
                )
            chain = self.chains.get(prepared.chain_id)
            if chain is None or chain.reserved_by != prepared.request_id:
                raise TokenStateCorruptionError(
                    "Reserved chain is missing or owned by another request",
                    session_id=self.session_id,
                    request_id=prepared.request_id,
                )
            segment = chain.current_segment
            boundary_adjustment: BoundaryAdjustment | None = None
            if prepared.is_new_chain:
                if segment.exact_token_ids != prepared.context_ids:
                    raise TokenStateCorruptionError(
                        "New-chain token state changed before commit",
                        session_id=self.session_id,
                        request_id=prepared.request_id,
                    )
            else:
                if segment.exact_token_ids != prepared.exact_prefix_ids:
                    raise TokenStateCorruptionError(
                        "Existing-chain exact prefix changed before commit",
                        session_id=self.session_id,
                        request_id=prepared.request_id,
                    )
                removed_count = len(prepared.removed_prefix_ids)
                if removed_count:
                    if tuple(segment.response_ids[-removed_count:]) != prepared.removed_prefix_ids:
                        raise TokenStateCorruptionError(
                            "Continuous Token removal no longer matches the reserved segment tail",
                            session_id=self.session_id,
                            request_id=prepared.request_id,
                        )
                    removed_logprobs = (
                        list(segment.response_logprobs[-removed_count:])
                        if segment.response_logprobs is not None
                        else None
                    )
                    boundary_adjustment = BoundaryAdjustment(
                        reason="model_boundary_prefix_removal",
                        request_id=prepared.request_id,
                        turn_id=prepared.turn_id,
                        removed_prefix_token_ids=prepared.removed_prefix_ids,
                        removed_generation_mask=tuple(segment.response_generation_mask[-removed_count:]),
                        removed_logprobs=tuple(removed_logprobs) if removed_logprobs is not None else None,
                        removed_source_turn_ids=tuple(segment.source_turn_ids[-removed_count:]),
                        removed_provenance_kinds=tuple(segment.provenance_kinds[-removed_count:]),
                    )
                    del segment.response_ids[-removed_count:]
                    del segment.response_generation_mask[-removed_count:]
                    if segment.response_logprobs is not None:
                        del segment.response_logprobs[-removed_count:]
                    del segment.source_turn_ids[-removed_count:]
                    del segment.provenance_kinds[-removed_count:]
                    segment.boundary_adjustments.append(boundary_adjustment)
                segment.response_ids.extend(prepared.appended_context_ids)
                segment.response_generation_mask.extend([0] * len(prepared.appended_context_ids))
                if segment.response_logprobs is not None:
                    segment.response_logprobs.extend([0.0] * len(prepared.appended_context_ids))
                segment.source_turn_ids.extend([None] * len(prepared.appended_context_ids))
                segment.provenance_kinds.extend(["replayed_context"] * len(prepared.appended_context_ids))
                if segment.exact_token_ids != prepared.context_ids:
                    raise TokenStateCorruptionError(
                        "Continuous Token boundary adjustment does not match the reserved backend prompt",
                        session_id=self.session_id,
                        request_id=prepared.request_id,
                    )

            segment.response_ids.extend(output.token_ids)
            segment.response_generation_mask.extend([1] * len(output.token_ids))
            if segment.response_logprobs is not None:
                if output.logprobs is None:
                    segment.response_logprobs = None
                else:
                    segment.response_logprobs.extend(output.logprobs)
            segment.source_turn_ids.extend([prepared.turn_id] * len(output.token_ids))
            segment.provenance_kinds.extend(["generated"] * len(output.token_ids))
            segment.validate()

            chain.message_history = [*prepared.prompt.messages, response_message]
            chain.message_prefix_hashes = [
                prefix.rolling_hash for prefix in self.canonicalizer.prefix_hashes(chain.message_history)
            ]
            turn_metadata = dict(metadata or {})
            turn_metadata.update(output.metadata)
            if not prepared.is_new_chain and boundary_adjustment is not None:
                turn_metadata["continuous_token_boundary_adjustment"] = boundary_adjustment.to_dict()
            turn = TurnRecord(
                turn_id=prepared.turn_id,
                source_turn_id=prepared.turn_id,
                chain_id=chain.chain_id,
                segment_id=segment.segment_id,
                request_messages=prepared.prompt.messages,
                request_tools=prepared.prompt.tools,
                tools_fingerprint=prepared.prompt.tools_fingerprint,
                backend_prompt_ids=prepared.context_ids,
                output_token_ids=output.token_ids,
                output_logprobs=output.logprobs,
                generation_mask=tuple(1 for _ in output.token_ids),
                routed_experts=output.routed_experts,
                response_message=response_message,
                finish_reason=output.finish_reason,
                upstream_replica=self.upstream_replica,
                delivery_status=delivery_status,
                metadata=turn_metadata,
            )
            chain.turns.append(turn)
            chain.state = ChainLifecycleState.ACTIVE
            chain.reserved_by = None
            chain.pending_request_messages = None
            chain.updated_sequence = self._next_sequence_locked()
            self._complete_reservation_locked(
                prepared,
                actual_token_count=len(prepared.context_ids) + len(output.token_ids),
            )
            self._record_event_locked(
                "request_committed",
                request_id=prepared.request_id,
                chain_id=chain.chain_id,
                metadata={
                    "output_token_count": len(output.token_ids),
                    "delivery_status": delivery_status.value,
                    "removed_prefix_token_count": len(prepared.removed_prefix_ids),
                },
            )
            self._condition.notify_all()
            return turn

    def _complete_reservation_locked(
        self,
        prepared: PreparedTurn,
        *,
        actual_token_count: int,
    ) -> None:
        self._in_flight.pop(prepared.request_id, None)
        self._reserved_token_count -= prepared.estimated_token_budget
        self.total_token_count += actual_token_count

    async def fail_turn(
        self,
        prepared: PreparedTurn,
        *,
        failure_reason: str,
        upstream_called: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        async with self._condition:
            registered = self._in_flight.get(prepared.request_id)
            if registered != prepared:
                return
            chain = self.chains.get(prepared.chain_id)
            if chain is not None and chain.reserved_by == prepared.request_id:
                chain.reserved_by = None
                chain.pending_request_messages = None
                if prepared.is_new_chain and not chain.turns:
                    has_children = any(
                        child.parent_chain_id == chain.chain_id
                        for child in self.chains.values()
                        if child.chain_id != chain.chain_id
                    )
                    if has_children:
                        chain.state = ChainLifecycleState.COMPLETED
                    else:
                        self.chains.pop(chain.chain_id, None)
                else:
                    chain.state = ChainLifecycleState.ACTIVE
            self._in_flight.pop(prepared.request_id, None)
            self._reserved_token_count -= prepared.estimated_token_budget
            if upstream_called:
                self.total_token_count += len(prepared.context_ids)
            else:
                self.model_request_count -= 1
            self._record_event_locked(
                "request_failed",
                request_id=prepared.request_id,
                chain_id=prepared.chain_id,
                failure_reason=failure_reason,
                metadata=metadata,
            )
            self._condition.notify_all()

    async def mark_delivery(
        self,
        *,
        turn_id: str,
        status: DeliveryStatus,
    ) -> None:
        async with self._condition:
            for chain in self.chains.values():
                for index, turn in enumerate(chain.turns):
                    if turn.turn_id == turn_id:
                        chain.turns[index] = replace(turn, delivery_status=status)
                        self._record_event_locked(
                            "delivery_updated",
                            chain_id=chain.chain_id,
                            metadata={"turn_id": turn_id, "delivery_status": status.value},
                        )
                        return
            raise SessionConflictError(
                f"Unknown turn {turn_id}",
                session_id=self.session_id,
            )

    async def begin_finalization(self, *, timeout_seconds: float) -> tuple[ChainState, ...]:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        async with self._condition:
            if self.state is SessionLifecycleState.FINALIZED:
                return tuple(self.chains.values())
            if self.state is SessionLifecycleState.ABORTED:
                raise SessionInactiveError(
                    f"Session {self.session_id} is aborted",
                    session_id=self.session_id,
                )
            if self.state is SessionLifecycleState.ACTIVE:
                self.state = SessionLifecycleState.FINALIZING
                self._record_event_locked("finalization_started")
            try:
                await asyncio.wait_for(
                    self._condition.wait_for(lambda: not self._in_flight),
                    timeout=timeout_seconds,
                )
            except TimeoutError as exc:
                raise SessionConflictError(
                    "Timed out waiting for in-flight Proxy requests",
                    session_id=self.session_id,
                    details={"in_flight_request_ids": list(self._in_flight)},
                ) from exc
            return tuple(self.chains.values())

    async def complete_finalization(self, bundle: TrajectoryBundle) -> TrajectoryBundle:
        async with self._condition:
            if self._finalized_bundle is not None:
                return self._finalized_bundle
            if self.state is not SessionLifecycleState.FINALIZING:
                raise SessionConflictError(
                    "Session is not finalizing",
                    session_id=self.session_id,
                )
            if bundle.session_id != self.session_id:
                raise ValueError("finalized bundle belongs to another session")
            self._finalized_bundle = bundle
            self.state = SessionLifecycleState.FINALIZED
            for chain in self.chains.values():
                if chain.state is ChainLifecycleState.ACTIVE:
                    chain.state = ChainLifecycleState.COMPLETED
            self._record_event_locked(
                "finalization_completed",
                metadata={"trajectory_count": len(bundle.trajectories)},
            )
            return bundle

    async def cached_finalized_bundle(self) -> TrajectoryBundle | None:
        async with self._condition:
            return self._finalized_bundle

    async def abort(self) -> None:
        async with self._condition:
            if self.state in {SessionLifecycleState.ABORTED, SessionLifecycleState.FINALIZED}:
                return
            self.state = SessionLifecycleState.ABORTED
            self._record_event_locked("session_aborted")
            self._condition.notify_all()


class ProxySessionManager:
    """Experiment-scoped in-memory registry with immutable sticky replicas."""

    def __init__(
        self,
        *,
        replica_endpoints: Sequence[str],
        frontend_base_url: str,
        continuous_token_codec: ContinuousTokenCodec,
        canonicalizer: Canonicalizer | None = None,
        model_name: str | None = None,
        replica_picker: Callable[[Sequence[str]], str] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        endpoints = tuple(replica_endpoints)
        if not endpoints or any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoints):
            raise ValueError("replica_endpoints must contain at least one non-empty URL")
        if len(set(endpoints)) != len(endpoints):
            raise ValueError("replica_endpoints must not contain duplicates")
        if not isinstance(frontend_base_url, str) or not frontend_base_url:
            raise ValueError("frontend_base_url must be a non-empty URL")
        self.replica_endpoints = endpoints
        self.frontend_base_url = frontend_base_url.rstrip("/")
        self.continuous_token_codec = continuous_token_codec
        self.canonicalizer = canonicalizer or Canonicalizer()
        self.model_name = model_name
        self._replica_picker = replica_picker or secrets.choice
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self._sessions: dict[str, ProxySession] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        *,
        session_id: str,
        generation_spec: GenerationSpec,
    ) -> AgentSessionHandle:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                if existing.generation_spec != generation_spec:
                    raise SessionConflictError(
                        "Session ID already exists with a different generation spec",
                        session_id=session_id,
                    )
                return self._handle(existing)
            session = ProxySession(
                session_id=session_id,
                auth_token=self._token_factory(),
                generation_spec=generation_spec,
                upstream_replica=self._replica_picker(self.replica_endpoints),
                model_name=self.model_name,
                continuous_token_codec=self.continuous_token_codec,
                canonicalizer=self.canonicalizer,
            )
            self._sessions[session_id] = session
            return self._handle(session)

    def _handle(self, session: ProxySession) -> AgentSessionHandle:
        return AgentSessionHandle(
            session_id=session.session_id,
            frontend_endpoint=f"{self.frontend_base_url}/sessions/{session.session_id}",
            auth_token=session.auth_token,
            model_name=session.model_name,
        )

    async def get_session(self, session_id: str) -> ProxySession:
        async with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise UnknownSessionError(
                    f"Unknown Proxy session {session_id}",
                    session_id=session_id,
                ) from exc

    async def authorize(self, session_id: str, token: str | None) -> ProxySession:
        session = await self.get_session(session_id)
        if token is None or not secrets.compare_digest(session.auth_token, token):
            raise InvalidSessionTokenError(
                "Invalid or missing Proxy session token",
                session_id=session_id,
            )
        await session.require_active()
        return session

    async def abort_session(self, session_id: str) -> None:
        session = await self.get_session(session_id)
        await session.abort()

    async def close(self) -> None:
        async with self._lock:
            sessions = tuple(self._sessions.values())
        await asyncio.gather(*(session.abort() for session in sessions))
