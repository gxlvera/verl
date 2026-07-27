# Agent Service integration

This directory separates the transport-neutral Driver SDK, the verl-specific
integration, and server-side Agent Service components.

## Package layout

- `client_sdk/` contains `TransportClient`, `RayTransportClient`,
  `AgentExecutor`, and the Task wire models. It has no verl dependency.
- `verl_adapter/` contains `VerlAgentServiceRuntime`, `RolloutAdapter`, and
  validation for verl's Agent Service configuration.
- `proxy/` contains the experiment-scoped HTTP Proxy, task-scoped sessions,
  canonical routing, Continuous Token adapter, upstream client, and trajectory
  materializer.
- `trajectory_selection/` contains the AgentService-side selector registry,
  built-in `all`/`longest` strategies, and complete-bundle retention helper.
- `execution_backend/` contains the controller-facing execution contract and
  the V0 `LocalExecutionBackend` implementations.
- `agent_task_controller/` is the package for the server-side Agent Task
  Controller implementation.

The V0 Controller and Store can be added under their server-side packages
without mixing them into the Driver SDK or verl adapter.

## Local execution backend

`LocalExecutionBackend` is selected once for an Agent Service instance and
owns the private `session_id -> native runtime handle` mapping. It exposes the
async `start`, `launch`, `inspect`, `wait`, `cancel`,
`execute_in_environment`, `cleanup`, and `close` contract. Launch is
idempotent for the same session and resolved `TaskExecutionSpec`; reusing a
session with a different spec is rejected.

Two local runtime modes are available:

- `coroutine` starts a fixed number of persistent worker processes. Each
  worker hosts multiple white-box AgentLoop coroutines. An entrypoint is an
  importable async function accepting `TaskExecutionSpec`, or a zero-argument
  class with an async `run(TaskExecutionSpec)` method.
- `process` starts one subprocess for each black-box command Task. The Backend
  owns its process group, exit status, stdout/stderr, timeout, and cancellation.

Concurrency and queueing do not belong to the Backend or Driver SDK. The
future AgentTaskController will apply the experiment-level
`admission.max_concurrent_tasks` and `admission.max_queued_tasks` policy before
calling `launch`. The Driver may submit a complete rollout batch without a
second local in-flight limit.

## Runtime ownership

After `RayPPOTrainer.init_workers()`, `TaskRunner` starts one Agent Service
runtime:

```python
agent_service_runtime = VerlAgentServiceRuntime(
    trainer=trainer,
    config=config,
)
agent_service_runtime.start()
rollout_adapter = agent_service_runtime.get_rollout_adapter(
    tokenizer=tokenizer,
    processor=processor,
)
```

`VerlAgentServiceRuntime.start()` creates the configured Ray actor internally,
waits for it to become ready, and installs `RayTransportClient` and
`AgentExecutor`. `get_rollout_adapter()` separately returns the adapter needed
by Trainer. The actor uses Ray's default owner-scoped lifecycle, and its raw
actor handle is private to the runtime. The transport client receives only a
JSON-compatible endpoint with the actor name and Ray namespace. It resolves the
handle internally and never passes the handle to Trainer, the rollout adapter,
or an Agent Service RPC.

Driver-to-service control-plane calls continue to use Ray actor RPC. Separately,
the Agent-facing Proxy starts an experiment-scoped FastAPI/uvicorn listener.
Each Task receives a session-scoped URL and opaque credential; the Agent never
receives rollout replica endpoints or credentials.

The configured `agent_service.ray_actor.actor_class` may be a plain Python class
or an existing Ray ActorClass. Its contract is:

```python
class AgentServiceActor:
    def __init__(self, startup_config: dict): ...
    def wait_ready(self, timeout_seconds: float) -> None: ...
    def submit(self, task_spec: dict, idempotency_key: str | None) -> dict: ...
    def get_status(self, task_id: str) -> dict: ...
    def wait_any(
        self,
        task_ids: list[str],
        timeout_seconds: float,
        max_results: int,
    ) -> dict: ...
    def cancel(self, task_id: str) -> dict: ...
    def stop(self, graceful_timeout_seconds: float) -> None: ...
```

`submit` returns `{"task_id": "..."}`; `get_status` and `cancel` return one
serialized `TaskSnapshot`; `wait_any` returns `{"tasks": [...]}`. Shutdown
first cancels outstanding Tasks through `AgentExecutor`, closes the transport
client, asks the actor to stop gracefully, and finally kills it with
`no_restart=True`.

### Proxy tokenizer bootstrap

At startup, `VerlAgentServiceRuntime` resolves
`config.actor_rollout_ref.model` into a plain JSON-compatible dictionary and
installs it as `startup_config["proxy"]["model_config"]`. It preserves the
original `path` and `tokenizer_path`, rather than passing TaskRunner's
node-local checkpoint copy or a tokenizer object.

The Proxy owns tokenizer construction. It may use the verl-independent helper:

```python
from agent_service import load_tokenizer_and_processor_from_model_config

model_config = startup_config["proxy"]["model_config"]
tokenizer, processor = load_tokenizer_and_processor_from_model_config(
    model_config,
    source_resolver=artifact_resolver.resolve,
)
```

`source_resolver` is optional when `tokenizer_path` (or its `path` fallback) is
already a local directory or a Hugging Face model ID. A remote service should
provide a resolver for HDFS or object-store locations. The helper imports no
verl modules and does not construct `HFModelConfig` or load model weights. It
returns `processor=None` for text-only models and initializes the supported
multimodal processor types using the same behavior as verl's legacy loader.

The concrete V0 Proxy is constructed with:

```python
from agent_service import build_hosted_proxy

proxy = build_hosted_proxy(startup_config)
await proxy.start()
```

It exposes OpenAI Chat Completions and Anthropic Messages, including SSE and
tool-use responses. OpenAI Responses remains deferred. The Proxy requires the
token-in/token-out `POST /agent_service/generate` upstream protocol so every request carries
the exact token IDs produced by Continuous Token.

Routing happens before chat-template rendering. The only lineage authority is
the canonical message prefix plus the canonical tool fingerprint. A same-chain
continuation reuses the stored assistant output token IDs directly. A trusted
Continuous Token builder may declare a model-specific tail-boundary removal
(for example GLM's ambiguous observation/user stop token); Proxy validates that
the retained prefix is byte-for-byte unchanged and records the removed token's
ID, logprob, mask, and provenance. Undeclared removal or any other prefix
rewrite still fails before inference.

## Driver data contract

The Task API uses Ray RPC with transport-neutral wire values:

- `submit(task_spec: dict, idempotency_key: str | None)`
- `get_status(task_id: str)`
- `wait_any(task_ids: list[str], timeout_seconds: float, max_results: int)`
- `cancel(task_id: str)`

The built-in `RolloutAdapter` creates one `TaskSpec` per `DataProto` sample and
restores service completion results to input order. Its default trajectory
decoder expects the token-level fields below:

```json
{
  "prompt_ids": [1, 2],
  "response_ids": [3, 4],
  "response_mask": [1, 1],
  "response_logprobs": [-0.1, -0.2],
  "num_turns": 2,
  "metrics": {},
  "extra_fields": {}
}
```

`loss_mask` is accepted as an alias for `response_mask`. A Task declares
`trajectory_selection` with default strategy `longest`. Proxy finalization
always returns and retains the complete bundle; AgentService applies the
trusted server-side selector afterward. If `all` or a custom selector returns
multiple trajectories, `RolloutAdapter` expands them into multiple output
batch rows while duplicating the corresponding sample metadata.

### Per-sample fields and inline multimodal data

TaskSpec separates the stable Agent Service control plane from verl's open-ended
dataset schema:

- `problem`, `agent`, `execution`, `environment`, `reward`, `generation`, and
  `lifecycle` are stable service-owned fields.
- `sample_fields` preserves per-sample `DataProto.non_tensor_batch` entries
  under their original field names, except `raw_prompt` and verl's private
  `__do_sample__` rollout override. `raw_prompt` is normalized once into
  `problem.messages`; all other sample fields remain the network equivalent of
  keyword arguments passed to a native `AgentLoop.run()`.
- Binary media stays at the same nested field path used by native verl. Even
  though Ray can serialize Python objects, the wire contract intentionally
  cannot carry `PIL.Image` or raw bytes. The value at that position is an inline
  object containing `kind`, `content_type`, `encoding=base64`, and `data`. No
  separate asset registry or `asset_id` lookup is required.

For interoperability with protocol-aware runtimes, `problem.messages` is the
only wire representation of the sample's `raw_prompt`. The sample value always
wins over a static `task.problem.messages` configuration. A runtime that needs
legacy verl behavior should hydrate the messages, set
`agent_kwargs["raw_prompt"]`, and pass the remaining `sample_fields` using their
original names.

Example:

```json
{
  "problem": {
    "messages": [{
      "role": "user",
      "content": [{
        "type": "image",
        "image": {
          "kind": "image",
          "content_type": "image/png",
          "encoding": "base64",
          "data": "..."
        }
      }]
    }]
  },
  "sample_fields": {
    "tools_kwargs": {"image_zoom": {}},
    "custom_dataset_field": {"kept": true}
  }
}
```

To enable the integration, configure at least:

Agent Service defaults live in
`verl/trainer/config/agent_service/agent_service.yaml`. The main
`ppo_trainer.yaml` only owns the `agent_service.enabled` switch.

```yaml
agent_service:
  enabled: true
  upstream_protocol: generate
  ray_actor:
    actor_class: your_package.AgentServiceActor
  execution_backend:
    kind: local
    runtime:
      kind: coroutine
      worker_processes: 4
  admission:
    max_concurrent_tasks: 512
    max_queued_tasks: 1024
  task:
    agent:
      artifact: ./agents/my_agent
      entrypoint: my_agent.agent_loop:run
      frontend_protocol: openai_chat_completions
    reward:
      reward_function:
        kind: binary
    trajectory_selection:
      strategy: longest
      config: {}
```

For a black-box local Agent, replace the runtime and Task launch fields:

```yaml
agent_service:
  execution_backend:
    kind: local
    runtime:
      kind: process
  admission:
    max_concurrent_tasks: 512
    max_queued_tasks: 1024
  task:
    agent:
      artifact: ./agents/my_agent
      frontend_protocol: openai_chat_completions
    execution:
      command: [python, main.py]
```
