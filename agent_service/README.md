# Agent Service integration

This directory separates the transport-neutral Driver SDK, the verl-specific
integration, and server-side Agent Service components.

## Package layout

- `client_sdk/` contains `TransportClient`, `RayTransportClient`,
  `AgentExecutor`, and the Task wire models. It has no verl dependency.
- `verl_adapter/` contains `VerlAgentServiceRuntime`, `RolloutAdapter`, and
  validation for verl's Agent Service configuration.
- `proxy/` contains Proxy-owned helpers. It currently provides the
  verl-independent tokenizer/processor loader.
- `agent_task_controller/` is the package for the server-side Agent Task
  Controller implementation.

The V0 Controller, execution backend, Store, and AgentRuntime implementations
can be added under their server-side packages without mixing them into the
Driver SDK or verl adapter.

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

V0 does not start FastAPI, uvicorn, or any other HTTP server in the Service
Actor. Driver-to-service calls use Ray actor RPC. The request and response
values are nevertheless restricted to JSON-compatible dictionaries, lists,
strings, numbers, booleans, and null so a future HTTP transport does not change
the `AgentExecutor.submit/get_status/wait_any/cancel` API used by Driver code.

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

The inference `upstream_protocol` is independent of the Agent's
`frontend_protocol`. Public frontends support OpenAI Chat Completions, OpenAI
Responses, and Anthropic Messages. Upstreams additionally support `generate`,
the token-in/token-out `POST /generate` protocol exposed by compatible
inference backends.

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

`loss_mask` is accepted as an alias for `response_mask`.

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
  ray_actor:
    actor_class: your_package.AgentServiceActor
  execution_backend:
    kind: local
  task:
    agent:
      artifact: ./agents/my_agent
      frontend_protocol: openai_chat_completions
    execution:
      command: [python, main.py]
    reward:
      reward_function:
        kind: binary
```
