# AgentService Proxy Implementation Plan

Status: Draft for implementation
Last updated: 2026-07-26
Source RFC: `【开源】Verl Agent Service RFC`, revision 3347

## 1. Goal

Implement the AgentService Proxy as:

- An experiment-scoped LLM gateway.
- A caller of the project-owned ViLa Continuous Token implementation.
- A recorder and materializer of every linear trajectory produced inside one
  task-scoped ProxySession.
- A protocol boundary that accepts plaintext OpenAI/Anthropic-compatible
  requests from an Agent while hiding upstream inference endpoints and
  credentials.

V0 must materialize and retain complete token-level trajectory data inside
AgentService. The Proxy returns the complete bundle to the AgentService
control plane; the Driver receives only the trajectories selected by its
declared server-side selection policy. V0 must not depend on a replay buffer.

## 2. Non-negotiable architecture decisions

### 2.1 Conversation routing is decided before chat-template rendering

The authoritative routing key is:

```text
canonical message prefix
+
canonical tool-schema fingerprint
```

Here, `canonical messages` means the structured internal message list after an
OpenAI/Anthropic adapter has normalized the wire request, but before
`apply_chat_template`.

The routing pipeline is:

```text
OpenAI / Anthropic request
        |
        v
provider adapter
        |
        v
canonical messages + canonical tools
        |
        v
message-prefix and tool-context routing
        |
        v
select existing chain or create a new chain
        |
        v
ViLa ContinuousTokenBuilder
        |
        v
upstream LLM generation
```

The Proxy must not use a newly rendered chat-template string or newly
re-tokenized full prompt to decide conversation lineage.

### 2.2 Chat templates are a codec concern, not a lineage oracle

`apply_chat_template` is allowed only after a chain has been selected:

- A new chain uses full encoding with messages and tools.
- An existing chain reuses its exact stored token state and encodes only the
  new non-assistant suffix through ViLa `ContinuousTokenBuilder`.

The following is explicitly forbidden:

```text
re-render full request
    -> compare rendered/token prefix
    -> use that comparison to redefine conversation lineage
```

The reason is that `encode(decode(token_ids))` is not guaranteed to reproduce
the original token IDs, especially across assistant text, stop-token, and
tool-call round trips.

### 2.3 Use Miles-style online token continuity, not Slime-style post-hoc repair

For every linear chain, token continuity is constructed before the next LLM
inference:

```text
Turn 1:
    full_encode(messages_1, tools) -> P1
    backend(P1) -> A_raw
    persist exact P1 and exact A_raw

Turn 2:
    select chain by canonical messages + tools
    ViLa ContinuousTokenBuilder(
        exact_prefix=P1 + A_raw,
        new_non_assistant_suffix=delta(messages_1, messages_2),
    ) -> P2_continuous
    backend(P2_continuous) -> B_raw
```

The following invariants are mandatory:

1. Every turn preserves the exact prompt token IDs sent to the backend.
2. Every turn preserves the exact assistant output token IDs returned by the
   backend, together with aligned logprobs and generation metadata.
3. On the same linear chain, the current assistant output token IDs become the
   exact prefix of the next turn's backend prompt, except for a model-specific
   tail-boundary removal explicitly declared by the trusted ViLa builder.
4. The Proxy must not decode the assistant output and then re-tokenize it to
   build the next turn's prompt.
5. Only newly appended non-assistant context and the next assistant generation
   marker are encoded incrementally.
6. A model-specific builder may append boundary tokens after the stored prefix
   or remove a declared ambiguous boundary token from the tail (for example
   GLM's observation/user boundary). For a removal, the Proxy must verify that
   every retained prefix token is unchanged and preserve the removed token's
   original ID, logprob, generation mask, source turn, and provenance in
   boundary-adjustment metadata. It must not replace, re-tokenize, or modify
   any token outside the declared tail removal.

Slime-style processing is explicitly rejected:

```text
full-encode every turn independently
    -> generate with independently encoded P2
    -> classify CLEAN / REALIGN / FORK after generation
    -> overwrite the latest generated response during REALIGN
```

In particular, the length of a later assistant output must never determine
whether an earlier assistant output is retained for learning.

### 2.4 Lineage and token segments are separate concepts

```text
Lineage:
    The parent/branch relationship derived from canonical messages and tools.

Token segment:
    A range that can be represented as one exact continuous token sequence.
```

If message/tool routing matches but continuous-token merge fails:

- Do not reroute the request by rendered-prefix matching.
- Keep the same conversation lineage.
- In V0, fail the request before calling the upstream LLM and leave the chain
  state unchanged.
- Do not silently full-encode the request as a fallback, because that would
  violate the exact-prefix invariant above.
- Record `failure_reason="continuous_token_merge_failed"` and boundary
  diagnostic metadata on the failed request/session event.
- Do not set `split_reason` and do not create a new chain, token segment, or
  trajectory, because no split occurred in V0.

A trusted builder's declared tail-boundary removal is a successful merge, not
a merge failure. It is allowed only when:

- `removed_prefix_token_count` is non-negative and within the stored runtime
  token range;
- removal does not reach into the immutable initial prompt;
- the returned context begins with the complete retained prefix;
- the removed tail token IDs exactly match the stored segment tail; and
- the removed tokens remain losslessly available in raw turn and boundary
  metadata even when they are absent from the normalized runtime sequence.

An undeclared removal, an out-of-range removal, or any rewrite outside the
declared tail boundary remains a fail-closed merge error before inference.

An explicitly requested future mode may create a new token segment, but it must
preserve the old segment losslessly and expose the discontinuity to the caller.
It is not the V0 default.

### 2.5 Tool schemas participate directly in chain selection

Tools are request-level prompt context, not fake system messages. The effective
chain compatibility key is:

```python
ChainCompatibilityKey(
    tools_fingerprint=...,
    model_identity=...,
    tokenizer_revision=...,
    chat_template_revision=...,
    template_kwargs_fingerprint=...,
)
```

In V0, model, tokenizer, chat template, and template kwargs should be fixed for
the lifetime of a ProxySession. Chain selection therefore normally needs to
compare only the tool fingerprint plus the message prefix.

If tools change while messages remain the same, create a sibling chain and
record `split_reason="tools_changed"`.

### 2.6 Keep every trajectory

V0 currently plans to return multiple trajectories, following Uni-Agent's
multiple-active-chain model rather than selecting a single winning or longest
trajectory.

The Proxy must retain all active and completed chains that have been created.
Finalization must not:

- Select only the longest chain.
- Select only the most recently updated chain.
- Drop a subagent, compaction, or repeated-prompt sibling.

`Proxy.finalize_session()` returns AgentService a `TrajectoryBundle` containing
every materialized trajectory and enough lineage metadata to reconstruct their
relationships.

Collection/finalization and selection are separate stages:

```text
Driver declares TrajectorySelectionSpec
        |
        v
AgentService stores and validates the declaration
        |
        v
Proxy.finalize_session() -> complete TrajectoryBundle
        |
        v
AgentService invokes its server-side TrajectorySelector
        |
        v
selected trajectories only -> Driver
```

Proxy collection and `Proxy.finalize_session()` always return every trajectory
to AgentService. The selector is implemented and executed in AgentService, not
in the Driver and not in Proxy. AgentService uses it to create an ordered
selected bundle without mutating the complete raw `TrajectoryBundle`.

The selector contract is conceptually:

```python
class TrajectorySelector(Protocol):
    def select(
        self,
        bundle: TrajectoryBundle,
        context: TrajectorySelectionContext,
    ) -> list[str]:
        """Return an ordered, non-empty list of trajectory IDs from bundle."""
```

AgentService V0 must provide a selector registry/factory that supports built-in
names and server-registered custom implementations. A Driver declares only a
strategy name plus serializable configuration in
`TrajectorySelectionSpec`; it must not supply an arbitrary Python import path
or executable selector code. A selector may inspect trajectory metadata,
lineage, available rewards, token counts, and task/session metadata, but it
must not mutate, delete, or rewrite any trajectory.

V0 includes two built-in strategies:

- `all`: select every trajectory in deterministic materialization order.
- `longest`: select exactly one trajectory. Rank primarily by the number of
  tokens actually generated by the model, `sum(generation_mask)`, rather than
  total sequence length including replayed context. Deterministic tie-breakers
  are response token count, turn count, and materialization order.

The V0 default is `trajectory_selection="longest"`. `all` or a
server-registered custom selector can be declared by the Driver without
changing Proxy collection. Invalid selector output, including unknown IDs or
an empty selection for a non-empty bundle, must fail the AgentService
finalization/result step explicitly and leave the complete bundle unchanged.

AgentService must retain the complete bundle according to the task-result
artifact/retention policy even though the normal Driver response contains only
the selected trajectories. The result metadata must record the selector name
and version, configuration fingerprint, candidate trajectory IDs, selected
trajectory IDs, and selection diagnostics.

This mirrors Uni-Agent's separation: Gateway finalization returns all active
chains, while its Claude Code runner configures
`trajectory_selection="longest"` before reward processing and TransferQueue
writes. Uni-Agent's framework default is otherwise `"all"`. AgentService V0
chooses `"longest"` as its configurable default rather than making it an
inherent Proxy/Gateway constraint.

Retry and rollback semantics are intentionally not decided yet:

> 这个我还没有想好该怎么做。

In particular, this plan does not yet decide whether a retry should resume an
existing chain, create a copy-on-write sibling, materialize the abandoned
branch first, or use another representation. The most likely primary reference
is Uni-Agent Gateway, including its retry candidate selection and
multiple-chain fallback behavior. This is still a reference direction rather
than a decision to copy Uni-Agent's destructive latest-assistant rollback
verbatim.

### 2.7 Preserve generation facts independently of training policy

The Proxy must preserve:

- Exact input token IDs sent to the backend.
- Exact output token IDs returned by the backend.
- Output log probabilities.
- Routed-expert/R3 metadata when available.
- Canonical request and response messages.
- Source turn ID and chain/segment provenance.
- Whether tokens were actually generated or replayed as context.

The Proxy must not irreversibly encode one universal shared-prefix training
policy into the raw record.

At minimum, distinguish:

```text
generation_mask:
    Historical fact: whether a token came from an actual generation event.

loss_mask / loss_weight:
    Materialization policy: whether and how strongly a trainer optimizes that
    token in a particular linear sample.
```

For duplicated shared-prefix tokens, materialization policies may include:

- `per_leaf`: train the shared token in every leaf.
- `by_source_occurrence`: train each physical generation occurrence once.
- `aggregate`: aggregate descendant advantages/rewards at turn level.
- `weighted_per_leaf`: retain every leaf but use fractional `loss_weight`.

V0 must retain `source_turn_id` and token provenance so this policy can be
changed without recollecting rollout data. With the default `longest`
selector, cross-leaf shared-prefix weighting does not affect the initial
training path. When `all` or a multi-trajectory custom selector is used, the
configured materialization policy controls shared-prefix masks/weights;
trajectory selection itself must not rewrite them. The policy must not be
buried in the branch manager.

## 3. V0 scope and boundaries

### In scope

- Experiment-scoped Proxy lifecycle.
- Task-scoped ProxySession lifecycle.
- Fixed upstream inference replica registry supplied at AgentService startup.
- Per-session sticky upstream selection.
- Anthropic Messages protocol.
- OpenAI Chat Completions protocol.
- OpenAI Responses protocol, if required by the first Agent integrations.
- Non-streaming and streaming responses.
- Tool-use requests and responses.
- Continuous-token construction using the project-owned ViLa library.
- Multiple chain retention.
- Complete token-level trajectory finalization.
- Server-side pluggable trajectory selection in AgentService, with built-in
  `all` and `longest`, server-registered custom selectors, and `longest` as
  the default. The Driver declares the strategy but does not execute it.
- In-memory session state for the initial implementation.
- Proxy-only smoke tests with a fake Agent and Claude Code.

### Out of scope for V0

- Dynamic upstream replica registration/removal.
- Automatic rerouting after a sticky replica becomes unavailable.
- Replay Buffer integration.
- Cross-experiment Proxy sharing.
- Durable multi-process session recovery.
- Trainer-specific reward/advantage computation inside Proxy.
- A generalized replacement for the ViLa continuous-token library.

## 4. Proposed package layout

```text
agent_service/
├── trajectory_selection/
│   ├── __init__.py
│   ├── base.py
│   ├── builtins.py
│   └── registry.py
└── proxy/
    ├── __init__.py
    ├── models.py
    ├── canonical.py
    ├── routing.py
    ├── session.py
    ├── continuous_tokens.py
    ├── codec.py
    ├── upstream.py
    ├── materializer.py
    ├── server.py
    ├── client.py
    └── adapters/
        ├── __init__.py
        ├── base.py
        ├── anthropic.py
        ├── openai_chat.py
        └── openai_responses.py
```

Responsibilities:

- `trajectory_selection/base.py`: AgentService-side selector protocol and
  selection context/result contracts.
- `trajectory_selection/builtins.py`: V0 `all` and `longest` selectors.
- `trajectory_selection/registry.py`: trusted server-side selector
  registration and resolution from Driver declarations.
- `models.py`: public and internal dataclasses/enums.
- `canonical.py`: provider-independent message/tool normalization and stable
  fingerprints.
- `routing.py`: message-prefix tree/index and chain-selection rules.
- `session.py`: ProxySession lifecycle, concurrency, chain state, and commit.
- `continuous_tokens.py`: thin V0 wrapper around ViLa Continuous Token.
- `codec.py`: full/incremental template encoding and response decoding.
- `upstream.py`: sticky inference replica selection and generation client.
- `materializer.py`: chain/segment to `TrajectoryBundle` conversion.
- `server.py`: HTTP server, authentication, routing, streaming, and errors.
- `client.py`: control-plane client when AgentService and Proxy are in separate
  processes.
- `adapters/*`: wire-protocol translation only.

## 5. Data contracts

Define the minimal common contracts before implementing the server.

### 5.1 Public control-plane types

```python
@dataclass(frozen=True)
class AgentSessionHandle:
    session_id: str
    frontend_endpoint: str
    auth_token: str
    model_name: str | None


@dataclass(frozen=True)
class GenerationSpec:
    sampling_params: Mapping[str, Any]
    max_new_tokens: int
    max_model_requests: int
    max_total_tokens: int


@dataclass(frozen=True)
class TrajectorySelectionSpec:
    strategy: str = "longest"
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRunSpec:
    task_id: str
    task_spec: TaskSpec
    session_handle: AgentSessionHandle
    trajectory_selection: TrajectorySelectionSpec = field(
        default_factory=TrajectorySelectionSpec,
    )
```

Keep the Task/Backend contracts in their common package rather than under
`proxy/`:

- `AgentRunResult`
- `ArtifactRef`
- `TaskEvaluation`
- `TaskRunResult`
- `FinalReward`

### 5.2 Canonical prompt types

```python
@dataclass(frozen=True)
class CanonicalPrompt:
    messages: tuple[CanonicalMessage, ...]
    tools: tuple[CanonicalTool, ...]
    tools_fingerprint: str


@dataclass(frozen=True)
class MessagePrefix:
    message_count: int
    rolling_hash: str
```

Canonicalization rules must be versioned. Store `canonicalization_version` in
session/trajectory metadata.

Recommended rules:

- Normalize OpenAI and Anthropic messages into one internal schema.
- Normalize tool arguments represented as JSON strings versus objects.
- Remove explicitly wire-only fields such as transient tool-call IDs only when
  they are proven irrelevant to routing.
- Preserve all fields that can change prompt meaning.
- Preserve list order for messages, tools, tool calls, and schema arrays.
- Use stable JSON serialization for hashing.
- Store canonical structures in addition to hashes.

Hashes are indexes, not the semantic source of truth. After a hash match,
perform exact canonical structure comparison before reusing a chain.

### 5.3 Raw turn record

```python
@dataclass(frozen=True)
class TurnRecord:
    turn_id: str
    source_turn_id: str
    chain_id: str
    segment_id: str

    request_messages: tuple[CanonicalMessage, ...]
    request_tools: tuple[CanonicalTool, ...]
    tools_fingerprint: str

    backend_prompt_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...] | None
    generation_mask: tuple[int, ...]
    routed_experts: Any | None

    response_message: CanonicalMessage
    finish_reason: str
    upstream_replica: str
    delivery_status: str
    metadata: Mapping[str, Any]
```

`backend_prompt_ids` and `output_token_ids` are token truth. Do not reconstruct
them later from messages.

### 5.4 Chain and segment state

```python
@dataclass(frozen=True)
class BoundaryAdjustment:
    reason: Literal["model_boundary_prefix_removal"]
    request_id: str
    turn_id: str
    removed_prefix_token_ids: tuple[int, ...]
    removed_generation_mask: tuple[int, ...]
    removed_logprobs: tuple[float, ...] | None
    removed_source_turn_ids: tuple[str | None, ...]
    removed_provenance_kinds: tuple[str, ...]


@dataclass
class TokenSegmentState:
    segment_id: str
    prompt_ids: list[int]
    response_ids: list[int]
    response_generation_mask: list[int]
    response_logprobs: list[float]
    source_turn_ids: list[str]
    split_reason: str | None
    boundary_adjustments: list[BoundaryAdjustment]


@dataclass
class ChainState:
    chain_id: str
    parent_chain_id: str | None
    branch_message_index: int
    split_reason: str | None

    message_history: list[CanonicalMessage]
    message_prefix_hashes: list[str]
    tools: tuple[CanonicalTool, ...]
    tools_fingerprint: str

    segments: list[TokenSegmentState]
    turns: list[TurnRecord]
    state: str
    updated_sequence: int
```

### 5.5 Final output

```python
@dataclass(frozen=True)
class TokenProvenance:
    source_turn_id: str | None
    kind: Literal["prompt", "generated", "replayed_context"]


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    session_id: str
    chain_id: str
    segment_id: str
    parent_chain_id: str | None
    split_reason: str | None

    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    generation_mask: tuple[int, ...]
    loss_mask: tuple[int, ...] | None
    loss_weight: tuple[float, ...] | None
    response_logprobs: tuple[float, ...] | None
    token_provenance: tuple[TokenProvenance, ...]
    routed_experts: Any | None
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TrajectoryBundle:
    session_id: str
    trajectories: tuple[Trajectory, ...]
    lineage: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class TrajectorySelectionResult:
    selector_name: str
    selector_version: str
    config_fingerprint: str
    candidate_trajectory_ids: tuple[str, ...]
    selected_trajectory_ids: tuple[str, ...]
    selected_bundle: TrajectoryBundle
    diagnostics: Mapping[str, Any]
```

`Proxy.finalize_session()` returns the complete `TrajectoryBundle` internally
to AgentService. AgentService applies the stored `TrajectorySelectionSpec` and
places only `TrajectorySelectionResult.selected_bundle` in the normal
Driver-facing task result. The complete bundle remains an internal retained
artifact and is never mutated by selection.

## 6. Control-plane interface

```python
class Proxy(Protocol):
    async def start(self) -> None: ...

    async def create_session(
        self,
        *,
        session_id: str,
        generation_spec: GenerationSpec,
    ) -> AgentSessionHandle: ...

    async def finalize_session(
        self,
        session_id: str,
    ) -> TrajectoryBundle: ...

    async def abort_session(
        self,
        session_id: str,
    ) -> None: ...

    async def close(self) -> None: ...
```

Lifecycle:

```text
CREATED -> ACTIVE -> FINALIZING -> FINALIZED
                  \-> ABORTED
```

Required semantics:

- `create_session` is idempotent for the same session ID and same spec.
- Reusing a session ID with a different spec is rejected.
- Repeated `finalize_session` returns the same cached bundle.
- Repeated `abort_session` succeeds without side effects.
- New data-plane requests are rejected once finalization starts.
- Finalization waits for, cancels, or reports in-flight requests according to an
  explicit timeout policy.
- Unknown session, invalid token, malformed request, inactive session, upstream
  failure, and timeout use distinct error responses.

## 7. Data-plane interface

Recommended session-scoped endpoints:

```text
POST /sessions/{session_id}/v1/messages
POST /sessions/{session_id}/v1/chat/completions
POST /sessions/{session_id}/v1/responses
POST /sessions/{session_id}/v1/messages/count_tokens
```

The session path can be exposed as the Agent's protocol-compatible base URL.
The opaque `auth_token` must also authorize the session. Do not trust a body
field alone for session identity.

`count_tokens` is secondary. It should reuse the same adapter, canonicalizer,
and codec as generation requests.

## 8. Request processing algorithm

### Step 1: Authenticate and resolve session

- Resolve `{session_id}` from the route/base URL.
- Validate the opaque bearer/API-key token.
- Confirm the session is `ACTIVE`.
- Enforce request-count and total-token budgets.

### Step 2: Adapt the wire request

The adapter produces:

```python
InternalGenerationRequest(
    messages=...,
    tools=...,
    sampling_params=...,
    stream=...,
    protocol=...,
    request_metadata=...,
)
```

Adapters must not perform trajectory routing or mutate session state.

### Step 3: Canonicalize prompt context

- Canonicalize the internal messages.
- Canonicalize tools once.
- Use that same canonical tool representation for both fingerprinting and
  actual template rendering.
- Compute stable rolling message-prefix hashes.

### Step 4: Select a chain

Candidate filtering:

```text
candidate.tools_fingerprint == request.tools_fingerprint
```

Candidate ranking:

1. Existing message history is an exact canonical prefix of the request.
2. Prefer the deepest matching prefix.
3. For indistinguishable siblings, use a documented deterministic tie-breaker,
   such as most recently updated chain.
4. If a request repeats a completed prompt boundary without echoing the
   generated assistant, treat it as a new sibling sample rather than appending
   to the existing output.

If no candidate matches:

- Find the deepest canonical common message prefix for lineage metadata.
- Create a sibling/new root chain.
- Record one of:
  - `message_prefix_diverged`
  - `tools_changed`
  - `context_compaction`
  - `repeated_prompt_resample`
  - `ambiguous_parent`

Do not delete the old chain.

### Step 5: Build backend token input

New chain:

```python
prompt_ids = codec.encode_full(messages, tools)
```

Existing chain:

```python
merge_result = continuous_token_builder.merge(
    exact_prefix_token_ids=chain.current_segment.exact_token_ids,
    old_messages=chain.message_history,
    new_messages=request.messages,
    tools=request.tools,
)
```

The merge result must return:

- Exact `context_ids` for the backend.
- Tokens newly appended as non-trainable context.
- Any explicitly removed tail-boundary token IDs.
- Updated continuous-token state.
- Diagnostics and any boundary adjustment.

For the same chain without a declared removal, enforce:

```python
context_ids[:len(exact_prefix_token_ids)] == exact_prefix_token_ids
```

With a declared removal, enforce:

```python
retained = exact_prefix_token_ids[:-removed_prefix_token_count]
context_ids[:len(retained)] == retained
removed_token_ids == exact_prefix_token_ids[len(retained):]
```

`exact_prefix_token_ids` includes the prior turn's original assistant output
IDs. The builder must never obtain that prefix by decoding and re-tokenizing
the assistant message. The Proxy trims aligned runtime masks/logprobs only
after validating the declared removal, while the original `TurnRecord`
continues to preserve the physical generation fact.

On merge failure:

- Keep the same lineage.
- Fail before upstream inference.
- Leave messages, tokens, turns, and reservations unchanged.
- Record `failure_reason="continuous_token_merge_failed"` and relevant boundary
  diagnostics on the failure event; do not record a trajectory
  `split_reason`.
- Never fall back to independent full-prompt encoding on the same chain.

### Step 6: Call the sticky upstream replica

- The ProxySession selects one replica at creation.
- All requests in that session use the selected replica in V0.
- Forward token IDs and sanitized sampling parameters.
- Request output logprobs and routed-expert metadata when configured.
- Map connection errors to 502 and generation timeouts to 504.
- Do not silently pick another replica in V0.

### Step 7: Decode and respond

- Preserve raw output token IDs before decoding.
- Decode into the provider-neutral assistant message.
- Adapt it to the original wire protocol.
- Support both non-streaming and streaming event shapes.

For streaming, aggregate exact backend token metadata independently of how text
chunks are emitted to the client.

### Step 8: Commit atomically

After backend generation succeeds:

- Append output tokens, logprobs, masks, and routed-expert metadata.
- Append the canonical assistant message to chain history.
- Update rolling message-prefix hashes.
- Store `TurnRecord`.
- Release any chain reservation.

Track client delivery separately:

- `delivered`: response flush completed.
- `failed`: the client disconnected before delivery.
- `unknown`: delivery acknowledgement is impossible.

Never discard the generation fact merely because delivery failed. Whether an
undelivered turn is exposed for training is a materialization policy.

## 9. Concurrency model

Correctness-first V0 options:

### Option A: Serialize an entire session

- Hold one session lock across prepare, generation, and commit.
- Simplest correctness model.
- Can block parallel subagents sharing a session.

### Option B: Reserve chains, generate outside the lock

1. Under the session lock, select and reserve a chain or provisional sibling.
2. Release the lock during upstream generation.
3. Reacquire the lock and commit only to the reserved chain.
4. Release reservations on success, error, timeout, or cancellation.

Recommended V0: implement Option B if Claude Code/subagents issue concurrent
requests in one session; otherwise begin with Option A and keep the session API
compatible with later reservations.

The selected option must have tests for concurrent identical prompts,
concurrent siblings, finalization during generation, and upstream failure.

## 10. Materialization rules

Materialization traverses every chain and segment; it never chooses only one
leaf.

For every output token, preserve:

```text
source_turn_id
generation fact
replayed-context fact
logprob provenance
branch/segment provenance
```

Do not infer original tokens by re-running a tokenizer.

For the trajectories selected by AgentService, the V0 Driver contract should
either:

1. Receive raw generation/provenance plus a configured materialized
   `loss_mask/loss_weight`; or
2. Receive raw generation/provenance and let the rollout adapter materialize
   trainer-specific weights.

The final decision must be validated against verl/slime loss reduction and
advantage computation. Binary `train_once` is not sufficient when several leaf
rewards should jointly assign credit to one physical prefix action.

## 11. Error model

Suggested HTTP mapping:

| Condition | Status |
|---|---:|
| Invalid or missing session token | 401 |
| Unknown session | 404 |
| Malformed provider request | 400 |
| Unsupported protocol/model/parameter | 422 |
| Finalized/aborted/finalizing session | 409 |
| Session budget exceeded | 429 |
| Upstream connection/generation failure | 502 |
| Upstream timeout | 504 |
| Internal token-state corruption | 500 |

Every error should include:

- Stable error code.
- Session ID and request ID where safe.
- Retryability.
- Whether session/chain state was mutated.

## 12. Step-by-step implementation phases

### Phase 0: Repository audit and dependency pinning

Tasks:

- Inspect the existing `agent_service/proxy` tokenizer helper.
- Locate common Task/Backend data contracts.
- Pin the exact ViLa Continuous Token import, API, and version.
- Record supported tokenizer, processor, and chat-template revisions.
- Decide whether Proxy runs in-process or as a separate process in V0.

Exit criteria:

- A dependency map and import boundary are documented.
- No Proxy contract imports trainer-specific `DataProto` types.

### Phase 1: Define contracts and lifecycle

Tasks:

- Add public dataclasses and enums.
- Add `Proxy` protocol.
- Add session lifecycle and cached finalize result.
- Define error types.
- Define immutable startup replica registry and sticky picker.

Tests:

- Create/finalize/abort idempotency.
- Reject mismatched repeated create.
- Reject requests after finalization begins.
- Unknown session behavior.

Exit criteria:

- Control-plane unit tests pass without an HTTP server or LLM backend.

### Phase 2: Canonical adapters and fingerprints

Tasks:

- Implement provider-neutral internal messages.
- Implement Anthropic-to-internal translation.
- Implement OpenAI Chat-to-internal translation.
- Implement Responses-to-internal translation if included in V0.
- Implement versioned canonical message normalization.
- Implement canonical tool normalization/fingerprinting.
- Implement rolling message-prefix hashes plus structural confirmation.

Tests:

- Equivalent tool arguments represented as JSON string/dict.
- Wire-only IDs do not cause accidental split when configured as ignorable.
- Meaningful content/tool changes do split.
- Tool order and message order remain significant.
- Same messages with different tools cannot reuse a chain.

Exit criteria:

- Routing inputs are deterministic and independent of chat-template rendering.

### Phase 3: Multiple-chain session router

Tasks:

- Return multiple trajectories using a Uni-Agent-style multiple-active-chain
  session model.
- Implement chain/root/sibling state.
- Implement longest canonical prefix selection.
- Implement repeated-prompt sibling behavior.
- Implement deterministic ambiguity handling.
- Retain every old chain.
- Record parent IDs, branch index, and split reason.

Tests:

- Linear conversation remains one chain.
- Subagent system prompt creates a sibling and main chain can later resume.
- Context compaction creates a new chain.
- Edited earlier tool result creates a sibling.
- Repeated identical first prompt creates multiple siblings.
- All siblings survive finalization.

Exit criteria:

- Multiple-chain routing works using only canonical messages and tool context.

### Phase 4: Continuous-token integration

Tasks:

- Wrap ViLa Continuous Token behind a local protocol.
- Implement first-turn full encoding.
- Implement Miles-style, pre-inference suffix-only continuation.
- Feed the prior assistant's exact output token IDs directly into the next
  turn's prompt prefix.
- Encode only newly appended non-assistant messages and the next generation
  marker.
- Store exact prompt/output token truth for every turn.
- Implement merge diagnostics and strict, mutation-free failure.
- Accept trusted model-specific tail-boundary removals (not arbitrary prefix
  rewrites), align runtime metadata, and retain lossless boundary diagnostics.
- Validate logprob/token alignment.

Tests:

- Ordinary text continuation.
- `encode(decode(tokens)) != tokens` case: assert that Turn 2 backend input
  still contains Turn 1's original output IDs, not the re-encoded IDs.
- Tool-call parse/re-render round trip.
- Stop/EOS boundary behavior.
- GLM ambiguous boundary removal preserves the original generated token in the
  raw turn record while the next backend prompt uses the validated trimmed
  runtime prefix.
- Merge failure happens before inference and leaves the chain unchanged.
- A later short assistant response never masks or deletes an earlier long
  assistant response.
- No `CLEAN / REALIGN / FORK` post-hoc classifier exists in the inference
  trajectory path.

Exit criteria:

- Every successful continuation sends the stored generated prefix unchanged to
  the upstream backend, except for an explicitly declared and validated
  model-specific tail-boundary removal.
- No successful or failed continuation irreversibly replaces an earlier
  assistant's raw token IDs or logprobs; normalized runtime alignment may omit
  only a losslessly recorded declared boundary token.

### Phase 5: Upstream generation client

Tasks:

- Register immutable replica endpoints at Proxy startup.
- Select one sticky replica per session.
- Forward exact context token IDs and sampling parameters.
- Capture output token IDs, logprobs, finish reason, and routed experts.
- Add timeout, cancellation, and error translation.

Tests:

- Sticky routing.
- Replica failure does not silently reroute.
- Timeout and cancellation leave session state consistent.
- Missing/misaligned logprobs fail loudly.

Exit criteria:

- A fake token backend can drive a complete multi-turn ProxySession.

### Phase 6: Anthropic server path

Tasks:

- Implement `/v1/messages`.
- Implement non-streaming response serialization.
- Implement SSE streaming.
- Implement tool-use blocks and stop reasons.
- Resolve session from scoped base URL plus opaque credential.

Tests:

- Anthropic text generation.
- Streaming event order.
- Tool use and tool result.
- Invalid request/error response shapes.
- Client disconnect and delivery status.

Exit criteria:

- Anthropic SDK and a curl request work against the AgentService Proxy.

### Phase 7: OpenAI protocol paths

Tasks:

- Implement Chat Completions.
- Implement Responses if required.
- Reuse the same canonical request, router, token builder, and upstream client.
- Do not create protocol-specific trajectory stores.

Exit criteria:

- Equivalent Anthropic/OpenAI requests produce the same internal canonical
  state where semantics match.

### Phase 8: Trajectory materialization and AgentService selection

Tasks:

- Materialize all chains and token segments.
- Preserve lineage and per-token provenance.
- Add configurable shared-prefix `loss_mask/loss_weight` policy.
- Cache the finalized bundle.
- Validate every aligned array length.
- Implement the AgentService-side selector protocol, registry, `all`, and
  `longest`.
- Read `TrajectorySelectionSpec` from the Driver's task declaration, resolve
  only trusted server-registered selectors, and apply it after Proxy
  finalization.
- Return only the selected bundle in the normal Driver-facing result while
  retaining the complete bundle as an internal task-result artifact.
- Record complete selection provenance and diagnostics.

Tests:

- Two and three sibling trajectories are all returned.
- Shared generated turns retain stable `source_turn_id`.
- Replayed context is distinguishable from a newly generated token.
- Repeated finalize returns byte-equivalent content.
- No longest-only filtering exists inside Proxy finalization.
- The `all` selector causes AgentService to return every finalized trajectory
  to the Driver in deterministic order.
- The `longest` selector causes AgentService to return only the trajectory with
  the largest `sum(generation_mask)` to the Driver.
- Longest-selection tie-breakers are deterministic.
- A Driver-declared, server-registered custom selector is invoked through the
  same stable interface.
- A Driver cannot supply an arbitrary executable/import path.
- Invalid selector output fails without mutating the raw bundle.

Exit criteria:

- `Proxy.finalize_session()` returns a complete and deterministic
  `TrajectoryBundle`.
- AgentService returns only the selected bundle to the Driver.
- The complete bundle remains intact after any selected view is constructed.

### Phase 9: Proxy-only smoke tests

Run without TaskRunner:

```text
Proxy.start()
  -> Proxy.create_session()
  -> sandbox curl /v1/messages
  -> sandbox Claude Code
  -> Proxy.finalize_session()
  -> inspect TrajectoryBundle
```

Acceptance:

- Claude Code calls the AgentService Proxy, not a debug gateway.
- At least one multi-turn tool-use run yields exact token-level trajectory data.
- A forced branch preserves both trajectories.
- A forced tool-schema change creates a sibling.
- Streaming and non-streaming both record token truth.
- Finalize/abort lifecycle behavior is correct.

### Phase 10: Observability and hardening

Add metrics/logs for:

- Active sessions, chains, segments, and in-flight requests.
- Requests and tokens per session.
- Chain splits by reason.
- Continuous-token merge failures.
- Upstream latency/errors by replica.
- Finalized trajectory count and token count.
- Client disconnects.
- Session lifecycle transitions.

Logs must never expose auth tokens, upstream credentials, or hidden reward
fields.

## 13. Required test matrix

| Area | Required cases |
|---|---|
| Routing | append, sibling, new root, compaction, repeated prompt; retry/rollback deferred until its semantics are decided |
| Tools | same schema, renamed tool, changed parameters, order change |
| Tokens | exact-prefix continuation, decode/encode drift, stop boundary, tool rendering drift, mutation-free merge failure |
| Multiple chains | resume old chain, ambiguous sibling, retain every leaf |
| Training selection | Driver declares policy; Proxy returns all to AgentService; AgentService executes built-in `all`/`longest` or a server-registered custom selector; default `longest`; only selected trajectories return to Driver; deterministic order/tie-breaks; invalid output |
| Lifecycle | duplicate create, duplicate finalize, duplicate abort, post-finalize request |
| Concurrency | parallel siblings, reserved chain failure, finalize in flight |
| Protocol | Anthropic/OpenAI text, streaming, tools, malformed body |
| Upstream | timeout, 5xx, disconnect, missing logprobs |
| Materialization | provenance, masks/weights, alignment, deterministic ordering |

## 14. Reference implementation guidance

- Uni-Agent: primary reference for canonical message-prefix plus direct tool
  compatibility routing, multiple active chains, all-trajectory Gateway
  finalization, and downstream `trajectory_selection="longest"` used by its
  Claude Code RL recipe.
- Slime: reference only for protocol adapters, message-tree retention, and the
  distinction between semantic lineage and token-sample continuity. Do not use
  its post-hoc `CLEAN / REALIGN / FORK` token repair.
- Miles: primary behavioral reference for constructing the next backend prompt
  before inference by reusing the exact generated token prefix. Implement the
  concrete builder with ViLa Continuous Token rather than copying Miles'
  tokenizer classes verbatim.

Do not copy:

- Hard-coded longest-only filtering inside Proxy finalization.
- Any concrete retry/rollback policy before the open design item below is
  resolved.
- Tool-blind message routing.
- Rendered chat-template prefix as conversation lineage authority.
- Slime-style independent full encoding on every turn.
- Slime-style `REALIGN` that replaces a prior assistant span and zeros its
  loss/logprob data.
- Any threshold where a later output's length decides whether an earlier
  generated response survives.

## 15. Open decisions before coding

1. Exact ViLa Continuous Token API and pinned version.
2. Whether V0 supports OpenAI Responses or defers it after Anthropic and Chat
   Completions.
3. Session concurrency: full serialization versus chain reservations.
4. Delivery-failure materialization policy.
5. Default shared-prefix `loss_mask/loss_weight` policy if/when training on
   multiple trajectories is enabled after V0.
6. Exact routed-expert/R3 tensor ownership and serialization contract.
7. Whether V0 session state must survive a Proxy process restart.
8. Retry and rollback semantics: **这个我还没有想好该怎么做。** This
   includes whether retry is represented by resuming a chain, creating a
   sibling, preserving/materializing the abandoned branch, or another model.
   The likely primary reference is Uni-Agent Gateway, but whether to adopt,
   modify, or replace its destructive latest-assistant rollback remains open.

These decisions must not change the authoritative routing rule in Section 2.

## 16. Definition of done

Proxy V0 is complete when:

- It exposes the required standard plaintext Agent protocols.
- Every request is authenticated to exactly one task-scoped ProxySession.
- Session routing is sticky to one registered upstream replica.
- Canonical messages plus canonical tools are the sole lineage authority.
- ViLa Continuous Token constructs each continuation before inference by
  reusing the previous turn's exact assistant output IDs, with only trusted,
  validated model-specific tail-boundary removal allowed.
- The same-chain continuation path never full-re-tokenizes an echoed assistant
  message.
- A continuous-token merge failure occurs before inference and cannot mutate or
  erase prior token truth; a successful declared boundary removal retains the
  removed token losslessly in raw turn/boundary metadata.
- Every branch and token segment is retained.
- `Proxy.finalize_session()` returns every materialized trajectory to
  AgentService; AgentService executes the Driver-declared selector and returns
  only the selected trajectories to the Driver.
- AgentService supports built-in `all`, built-in `longest`, and trusted
  server-registered custom selectors, with `longest` as the default.
- Exact token IDs, logprobs, masks, routed experts, and provenance are
  available at finalization.
- Finalize and abort are idempotent under their documented semantics.
- Finalized sessions reject new requests.
- Unit, integration, streaming, tool-use, error, concurrency, and Claude Code
  smoke tests pass.
