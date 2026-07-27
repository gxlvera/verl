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

"""One shared request pipeline for every frontend wire protocol."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .codec import AssistantResponseDecoder
from .errors import MalformedRequestError, ProxyError, UpstreamGenerationError
from .models import CanonicalMessage, InternalGenerationRequest, TurnRecord
from .session import PreparedTurn, ProxySession
from .upstream import TokenGenerationOutput, TokenGenerationRequest, UpstreamGenerator


@dataclass(frozen=True)
class ProcessedGeneration:
    prepared: PreparedTurn
    output: TokenGenerationOutput
    turn: TurnRecord
    response_message: CanonicalMessage


class ProxyRequestProcessor:
    def __init__(
        self,
        *,
        upstream: UpstreamGenerator,
        response_decoder: AssistantResponseDecoder,
    ) -> None:
        self.upstream = upstream
        self.response_decoder = response_decoder

    async def generate(
        self,
        *,
        session: ProxySession,
        request_id: str,
        request: InternalGenerationRequest,
    ) -> ProcessedGeneration:
        try:
            prompt = session.canonicalizer.canonicalize(request)
        except (TypeError, ValueError) as exc:
            raise MalformedRequestError(
                "Request messages or tools are malformed",
                session_id=session.session_id,
                request_id=request_id,
                details={"error": str(exc)},
            ) from exc
        prepared = await session.prepare_turn(
            request_id=request_id,
            request=request,
            prompt=prompt,
        )
        sampling_params = dict(prepared.sampling_params)
        stop_token_ids = tuple(getattr(self.response_decoder, "stop_token_ids", ()))
        if stop_token_ids:
            sampling_params["stop_token_ids"] = sorted(
                set(sampling_params.get("stop_token_ids", ())) | set(stop_token_ids)
            )
        upstream_request = TokenGenerationRequest(
            request_id=request_id,
            session_id=session.session_id,
            prompt_token_ids=prepared.context_ids,
            sampling_params=sampling_params,
            model_name=session.model_name,
            metadata={"frontend_protocol": request.protocol.value},
        )
        try:
            output = await self.upstream.generate(
                replica_endpoint=session.upstream_replica,
                request=upstream_request,
            )
        except asyncio.CancelledError:
            await session.fail_turn(
                prepared,
                failure_reason="request_cancelled",
                upstream_called=True,
            )
            raise
        except ProxyError as exc:
            await session.fail_turn(
                prepared,
                failure_reason=exc.code,
                upstream_called=True,
                metadata=exc.details,
            )
            raise
        except Exception as exc:
            await session.fail_turn(
                prepared,
                failure_reason="upstream_generation_failed",
                upstream_called=True,
                metadata={"error": str(exc)},
            )
            raise UpstreamGenerationError(
                "Unexpected upstream generation failure",
                session_id=session.session_id,
                request_id=request_id,
                details={"error": str(exc)},
            ) from exc

        try:
            response_message = await self.response_decoder.decode(
                output,
                tools=[tool.to_dict() for tool in prompt.tools],
                request_id=request_id,
            )
            turn = await session.commit_turn(
                prepared,
                output=output,
                response_message=response_message,
                metadata={"frontend_protocol": request.protocol.value},
            )
        except asyncio.CancelledError:
            if prepared.request_id in session.in_flight_request_ids:
                fallback = CanonicalMessage.from_dict({"role": "assistant", "content": ""})
                await asyncio.shield(
                    session.commit_turn(
                        prepared,
                        output=output,
                        response_message=fallback,
                        metadata={"frontend_protocol": request.protocol.value, "response_decode_cancelled": True},
                    )
                )
            raise
        except Exception:
            # Generation has already happened. Preserve the reservation's
            # generation facts by committing a minimal assistant message when
            # only decoding failed, then surface the original error.
            if prepared.request_id in session.in_flight_request_ids:
                fallback = CanonicalMessage.from_dict({"role": "assistant", "content": ""})
                await session.commit_turn(
                    prepared,
                    output=output,
                    response_message=fallback,
                    metadata={"frontend_protocol": request.protocol.value, "response_decode_failed": True},
                )
            raise
        return ProcessedGeneration(
            prepared=prepared,
            output=output,
            turn=turn,
            response_message=turn.response_message,
        )
