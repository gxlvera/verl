# 【开源】Verl Agent Service RFC

## 背景与目标

目标是在代码上将 Agent rollout 系统和RL框架解耦，Agent Service的模块化设计让他自己成为一个闭环系统，也能让不同rl框架(verl/slime）能轻松接入。从易用性和可维护性出发，Agent Service部署实例与单次训练实验绑定，由 verl 在训练启动阶段拉起，并随该次 verl 运行结束而停止。

## 整体架构

虚线箭头：Agent Service 启动阶段的初始化与绑定；实线箭头：per Task 调用

图示语义：虚线启动配置在当前 V0 中对应 AgentServiceStartupConfig，其中携带 trainer.get_rollout_inference_addresses() 返回的固定 replica 地址列表；Proxy 启动时完成注册，proxy之后按照某种策略（e.g. Sticky session）给每个请求pick replica。

> **Whiteboard:** retained in the source Lark document.

## 组件职责

> 💡 **Version 0 核心边界**：Agent Service 在代码、API 和运行时契约上与 verl/slime 等训练后端解耦，但其部署实例与单次训练实验绑定：由 verl 在训练启动阶段拉起，并在 verl 正常结束或异常退出时停止。一个 Agent Service 实例只承载一个 Experiment，启动时创建一套逻辑 Proxy、ExecutionBackend、Store；每个 Task 由 ExecutionBackend 启动一个 Taskrunner (task runner里起agent）。V0 不提供 Replay Buffer， Agent Service需要返回完整trajectory数据给driver。

### 组件+生命周期

注：Driver侧每次 SubmitTask 对应一个sample的 rollout，并拥有独立的 task_id、session_id、ProxySession、与 AgentRuntime。Driver 如需重试，会重新提交一个新 Task，获得新的task_id, session_id等；Agent Service 不关联前后两次 Task。

| 组件 | 实例生命周期 | 创建与销毁时机 |
| --- | --- | --- |
| AgentTaskController | Experiment | 随 Agent Service 启动而创建，随 Agent Service 停止而销毁 |
| Proxy | Experiment | 启动时依据 AgentServiceStartupConfig 创建逻辑 Proxy，并注册固定的上游 LLM replica 地址列表；停止时关闭。一个逻辑 Proxy 可以扩展为多个 replica |
| ExecutionBackend | Experiment | 启动时依据启动配置选择并创建 Local / Ray / Sandbox / K8s Backend，停止时关闭 |
| TrajectoryStore | Experiment | 随 Agent Service 启动 / 停止 |
| ProxySession | Task 级 | Task 启动前创建；创建时随机选择并保存 selected_replica，trajectory finalize 后关闭 |
| AgentRuntime | Task 级 | ExecutionBackend.launch(session_id, ...) / cleanup(session_id) |

---

### AgentTaskController

Agent Service 的统一控制面。服务启动阶段由 bootstrap 注入并解析实验的启动配置；进入 READY 后，对外只接收 Task 级请求。

1. 启动阶段（非对外 API）：Service bootstrap 校验 AgentServiceStartupConfig，创建逻辑 Proxy，向 Proxy 注册 inference replica 地址列表，并创建选定的 ExecutionBackend与 Store。
2. Task 级请求：管理 Task 并维护唯一的 Task 状态机；为 Task 创建 ProxySession, 调用 ExecutionBackend 启动 AgentRuntime。Controller 本身不执行 AgentLoop/AgentRuntime，也不实现 Local/Ray/K8s 的具体调度逻辑。

**对外 API**

| 范围 | API | 职责 |
| --- | --- | --- |
| Task | Submit | 校验 TaskSpec，持久化 Task，创建 ProxySession，投影 TaskExecutionSpec，并调用当前 Agent Service 实例唯一的 ExecutionBackend；立即返回 task_id |
| Task | GetStatus | 按 task_id 立即返回 TaskSnapshot，不阻塞；用于状态查询、故障诊断和 Driver 恢复 |
| Task | WaitAny | 对一组 task_id 执行 long-poll；返回其中已进入终态的 0 到 max_results 个 TaskSnapshot，供 SDK 实现 as_completed |
| Task | CancelTask | 将取消意图写入状态机，并以当前 Task 调用本实例 ExecutionBackend.cancel |

**核心职责**

- 维护 Agent Service 实例生命周期：STARTING → READY → DRAINING → STOPPED；该生命周期由 launcher / 部署层驱动，不作为独立的 Experiment 管理 RPC 暴露。
- 维护唯一的 Task 状态机：CREATED → QUEUED → LAUNCHING → RUNNING → EXECUTION_FINISHED → FINALIZING_TRAJECTORY → COMPUTING_REWARD → CLEANING_UP → SUCCEEDED / FAILED / CANCELLED。Task 进入终态后不在服务端重试；Driver 如需重试，重新调用 SubmitTask。
- 持久化 Task 记录（含 TaskSpec）和状态迁移；不持久化或持有 Backend 原生 RuntimeHandle。内存对象只作为缓存，Store 才是恢复后的事实来源。
- 从 AgentServiceStartupConfig.admission 读取当前实验的最大并发度；Task 数量超过并发上限时进入队列并应用 backpressure。

⚠️ Agent 退出后先 finalize trajectory、执行 verifier 和 reward，再调用 Backend cleanup；不能因 Agent 进程退出就提前销毁 environment。

> 📌 **职责边界**：Controller 不转发 LLM 请求、不拼装 token 级 trajectory、不直接启动进程/Ray Actor/Pod

---

### Proxy

Agent Service 的 LLM Gateway , TITO (continuous token )调用者，与 trajectory recorder。Proxy 在启动时注册固定的上游 inference replica 地址列表，并为每个 ProxySession 选择一个 upstream replica。一个逻辑 Proxy 可以由多个 replica 共同实现（e.g. 4个proxy replica分别跑在4个独立进程上，每个session创建的时候随机选一个proxy 进程），并服务本实验的全部 Task。

- V0 启动时读取 AgentServiceStartupConfig.inference，注册 rollout LLM replica endpoint 列表、model alias、protocol 和 credential reference；该列表在服务运行期间不可更新。
- 向 AgentTaskController 提供 create_session(session_id) 和 finalize_trajectory(session_id) 接口。
- 对 AgentLoop 暴露公共明文协议：OpenAI Chat Completions、OpenAI Responses API、Anthropic Messages。
- 通过 agent http request中的某个字段识别 ProxySession；黑盒 Agent 可把 token 注入 API key / Authorization header，不修改标准请求 body。
- 做Tokenization，调Continuous Token来确保token-level no loss。
- 通过前缀匹配来识别出Multiple linear trajectory，记录每条Traj的 request、response、token id、logprob、loss mask, R3 expert。

> 💡 **V0 / V1 inference 路由**：V0 由 TaskRunner 在 Agent Service 启动前通过类似 `trainer.get_rollout_inference_addresses()` 的接口取得多个llm replica 地址，并随 AgentServiceStartupConfig 一次性传入。Proxy 为每个 session 的每次请求pick replica（e.g. 每个session pick一个replica，然后stick，或者每个session的每次请求load balance 打replica）。V0 不支持运行期间新增、删除或替换 replica，也不在选中地址失效后自动重新选择。V1 在 verl 侧增加 inference server router，Agent Service 启动时只接收稳定的 Router URL；Proxy 始终请求 Router，由 Router 负责动态 replica 注册、健康状态和路由选择。

---

### ExecutionBackend

> 💡 **定义**：ExecutionBackend 是 Agent Service 实例级（亦即单 Experiment 级）的 TaskRunner 调度抽象。它只决定 `TaskRunner.run()` 在哪里执行，不决定 Agent 最终在哪里运行。

**实例级配置**：Service bootstrap 在启动时根据 `AgentServiceStartupConfig.execution` 创建一种 Backend。本实例内的所有 Task 共用该 Backend，但可以选择不同的 TaskRunner 和 Agent。Backend kind 在实例运行期间不可变更。

| 实现 | 语义 |
| --- | --- |
| `LocalExecutionBackend` | 在 Agent Service 所在node上，把每个 TaskRunner 作为一个 coroutine / `asyncio.Task` 运行。Agent 是在当前 coroutine、子进程、容器还是 sandbox 中运行，由用户实现的 AgentRuntime 决定。 |
| `RayExecutionBackend` | 把 TaskRunner 调度到 Ray worker 或专用 Ray actor 上运行；一个 actor 可以并发承载多个 TaskRunner coroutine。Backend 只管理 Ray 原生执行 handle，Agent 的运行位置仍由 AgentRuntime 决定。 |
| `K8sExecutionBackend` | 为 TaskRunner 创建 Kubernetes Pod/Job，并在其中运行 TaskRunner 进程。Agent 可以与 TaskRunner 同 Pod、运行在 TaskRunner 创建的 sandbox 中，或访问外部 Environment；具体方式仍由 AgentRuntime 和 Environment 决定。 |

> 📌 **Sandbox 不是 ExecutionBackend kind**：sandbox 描述的是 TaskRunner 创建的 Environment 或 AgentRuntime 启动 Agent 的方式。Local、Ray、K8s 三种 Backend 中都可以使用 sandbox，因此 V0 不提供 `SandboxExecutionBackend`。

**调用方**：ExecutionBackend 由 AgentService 内部的单 Task orchestration 调用。如果后续将这段编排拆成独立 Controller，它也只是 AgentService 的内部控制面组件，不改变 Backend 的语义。

*ExecutionBackend 接口*
```python
@dataclass(frozen=True)
class TaskRunSpec:
    task_id: str
    task_spec: TaskSpec
    session_handle: AgentSessionHandle


class ExecutionBackend(Protocol):
    async def launch(
        self,
        session_id: str,
        spec: TaskRunSpec,
    ) -> None: ...

    async def wait(self, session_id: str) -> TaskRunResult: ...

    async def cancel(self, session_id: str) -> None: ...

    async def cleanup(self, session_id: str) -> None: ...
```

Backend 启动的是 TaskRunner，而不是 AgentRuntime。TaskRunner 在自己的 `run()` 中创建 Task-scoped Environment、调用 AgentRuntime、收集 artifacts、执行 Task-domain evaluation，并在返回前清理 Environment。

#### Session ID 作为统一执行标识

> 💡 **设计决策**：每个 Task 对应一个全局唯一的 `session_id`，同时索引该 Task 的 ProxySession、Backend execution 和 trajectory。Driver 重试时提交新 Task，并生成新的 `task_id` 与 `session_id`，不复用旧执行状态。

*Task 的统一标识*
```text
Task(task_id, session_id)
├── ProxySession
├── Backend Execution (TaskRunner)
├── AgentRuntime
└── Trajectory / Reward

ExecutionBackend private state:
session_id -> NativeRuntimeHandle
  Local: asyncio.Task
  Ray: ObjectRef / actor handle
  K8s: namespace + Pod/Job name
```

- AgentService 生成并持久化 `session_id`，创建 Proxy session，然后将包含 `AgentSessionHandle` 的 TaskRunSpec 交给 ExecutionBackend。
- ExecutionBackend 不向上层暴露原生运行 handle，只在内部维护 `session_id -> NativeRuntimeHandle`。
- `launch(session_id, spec)` 必须幂等：同一 session_id 与相同 spec 的重复调用复用已有执行；参数不一致时拒绝，不能启动第二份 TaskRunner。
- `wait(session_id)` 返回 TaskRunner 产生的权威 `TaskRunResult`，其中包含 AgentRunResult、artifacts、TaskEvaluation 和 metrics，但不包含权威 trajectory。
- `cancel(session_id)` 负责向 TaskRunner 传播取消，并等待其 `finally` cleanup 完成或记录 cleanup failure；随后 AgentService 才 abort Proxy session。
- Task-scoped Environment 的生命周期归 TaskRunner。ExecutionBackend 的 `cleanup()` 只清理 coroutine、Ray handle、Pod/Job 等 Backend-native 资源。
- ExecutionBackend 只报告执行事实，不负责创建/finalize/abort Proxy session，不执行 Task 重试，也不计算最终 RL reward。

*成功路径中的职责边界*
```text
AgentService
├── Proxy.create_session()
├── ExecutionBackend.launch(TaskRunSpec)
├── ExecutionBackend.wait() -> TaskRunResult
├── Proxy.finalize_session() -> TrajectoryBundle
├── RewardResolver.resolve(TaskEvaluation, TrajectoryBundle)
└── ExecutionBackend.cleanup()
```

---

### TaskRunner

> 💡 **定义**：TaskRunner 是单个 Task 的领域执行编排器。ExecutionBackend 决定 TaskRunner 在哪里运行；TaskRunner 在该执行位置创建 Task-scoped Environment、调用 AgentRuntime、收集产物并完成 Task-domain evaluation。

V0 提供 `BaseTaskRunner` 默认模板。用户可以通过覆写 hooks 实现 SWE、Search 等 Task；若默认模板不适用，也可以完整覆写 `run()`。

*BaseTaskRunner 默认模板*
```python
class BaseTaskRunner(ABC):
    async def run(self, spec: TaskRunSpec) -> TaskRunResult:
        async with self.create_environment(spec) as environment:
            await self.prepare_environment(spec, environment)
            agent_result = await self.run_agent(spec, environment)
            artifacts = await self.collect_artifacts(spec, environment, agent_result)
            evaluation = await self.evaluate(spec, environment, agent_result, artifacts)
            return TaskRunResult(
                agent_result=agent_result,
                artifacts=artifacts,
                evaluation=evaluation,
                metrics={},
            )
```

**主要扩展方法**

| 方法 | 职责 |
| --- | --- |
| `create_environment()` | 创建并持有本 Task 的 Environment，例如 SWE sandbox 或 Search 工具执行环境。 |
| `prepare_environment()` | 执行领域前处理；默认可为空。例如 SWE-Rebench 清理 Git 历史，Search 从 TaskSpec 读取 external dependency endpoint 并注入环境。 |
| `run_agent()` | 解析并调用 AgentRuntime，向其传入 Task 信息、Environment 和受限的 SessionHandle。 |
| `collect_artifacts()` | 收集 patch、agent log、final answer、citations 等领域产物。 |
| `evaluate()` | 执行 Task-domain evaluation，返回 TaskEvaluation；例如运行 SWE 测试或对 Search answer 做匹配。 |

**推荐继承关系**

```text
BaseTaskRunner
├── BaseSWETaskRunner
│   ├── SWEBenchTaskRunner
│   └── SWEReBenchTaskRunner
└── BaseSearchTaskRunner
    └── ASearcherTaskRunner
```

> 📌 **职责边界**：TaskRunner 不创建、finalize 或 abort ProxySession，不构造权威 token-level trajectory，也不计算最终 RL reward。外部服务由用户启动并随 TaskSpec 传入 endpoint；TaskRunner 只读取和使用这些信息，并只清理由自己创建的 Task-scoped Environment。

---

### AgentRuntime

Task 级的运行单元，由 ExecutionBackend 为每个 Task 创建。AgentRuntime 不是长期服务，也不是裸 AgentLoop。

- 接收由 TaskSpec 与当前 Task 上下文投影出的 TaskExecutionSpec，以及 Proxy endpoint/token；不得获得上游 replica 地址列表、inference credential、reward 私有配置或标准答案等不应暴露给 Agent 的字段。
- 白盒模式下加载并运行指定的 AgentLoop artifact；黑盒模式下启动binary。
- 为两种模式统一注入 OpenAI/Anthropic 兼容的 base URL 与凭证，使白盒和黑盒 Agent 都通过公共明文协议访问 Proxy。
- 注入 problem、environment 访问配置和任务资源，捕获异常、退出码、stdout/stderr、artifacts，并响应 timeout/cancel。

---

### Reward evaluator

负责在 Agent environment 仍存活时产出 Driver 可直接消费的最终 reward。

目前不需要把reward 也服务化，因为reward这块暂不需要独立部署和库容，且没有多个agent service用同一个reward engine需求。把用户传来的reward function、artifact （e.g. 单测脚本）交给ExecutionBackend，异步执行verify+计算final reward即可。

| 阶段 | 职责 |
| --- | --- |
|  |  |
| Verifier | 依据数据集固定规则判断结果是否正确。Coding 场景通常需要访问 Task 的 sandbox/container，因此 RewardEngine 将 VerifierSpec 转换为通用命令执行请求，并以 session_id 调用 ExecutionBackend.execute_in_environment(session_id, CommandExecutionSpec)。Backend 只执行命令并返回结果，不理解 verifier/reward 语义。 |
| Reward Mapper | 将 verifier 结果、trajectory 和 Task 元数据映射为算法用户定义的最终标量 reward。用户代码运行在隔离的 worker/runtime 中，而不是 Controller 主进程。 |

RewardEngine 返回 RewardResult，至少包含 reward标量（final reward + 其依赖的子reward，比如turn reward, tool call reward....）、verifier_result。Advantage、group normalization 等依赖 batch 的训练计算仍属于 Driver/Trainer。

---

### Driver 侧 SDK（AgentServiceClient / AgentExecutor）

Driver 侧 SDK 是 Agent Service 远程 API 的薄封装。V0 固定采用 `TaskFuture + as_completed` 编程风格，底层通过 `WaitAny` long-poll 获取完成状态和结果；不实现 callback，也不使用 Replay Buffer。SDK 连接的服务实例已经绑定当前实验，因此客户端请求不携带 experiment_id。

#### 服务端 API

*AgentTaskController 对外 API*
```python
submit(task_spec, idempotency_key=None) -> task_id
get_status(task_id) -> TaskSnapshot
wait_any(task_ids, timeout_seconds, max_results) -> list[TaskSnapshot]
cancel(task_id) -> TaskSnapshot
```

- `SubmitTask` 持久化 Task 后立即返回 `task_id`，不保持一个可能持续几十分钟的提交请求。
- `WaitAny` 批量等待一组 Task 中任意 Task 进入 terminal state，避免 SDK 对每个 Task 高频轮询；单个 `TaskFuture.result()` 可以通过 `WaitAny([task_id])` 实现，因此 V0 不必额外暴露 WaitTask。
- long-poll 在 timeout 内有 Task 完成时返回 1 到 `max_results` 个终态 TaskSnapshot；超时仍无任务完成时返回空列表，SDK 随后重新发起请求。
- V0 的最终响应直接携带 trajectory、final reward、和错误信息；以后结果过大时可以扩展为 result reference。

Agent Service 不执行 retry。Task 进入 `SUCCEEDED / FAILED / CANCELLED` 后，对应 Future 进入 terminal state；Driver 根据 TaskSnapshot 决定是否接受结果、丢弃结果或重新调用 SubmitTask 创建一个新 Task。

## 数据契约

### AgentServiceStartupConfig（启动配置，非 RPC）

这些字段属于当前训练实验，但不再通过独立的实验创建请求提交。它们主要来自 verl 配置文件；TaskRunner 在 trainer.init_workers() 后补充 rollout inference replica 地址，再由 AgentServiceLauncher 以启动参数或生成的配置文件传给 Agent Service 进程。V0 的 replica 列表在服务运行期间不可变；V1 会把它替换为稳定的 inference Router URL。AgentTaskController 不暴露 Experiment 创建或关闭 RPC。

| 字段 | 类型 | 内容与边界 |
| --- | --- | --- |
| `execution` | `ExecutionBackendSpec` | 必填。选择 Local / Ray / Sandbox / K8s，并包含对应的 cluster、namespace、runtime profile、资源和调度配置。一个 Agent Service 实例只选择一种 Backend，运行期间不可变更。 |
| `inference` | `InferenceSpec` | 必填（V0）。包含本实验启动时发现的 rollout LLM replica endpoint 列表、model alias、upstream protocol 和 credential reference。TaskRunner 在 trainer.init_workers() 后通过类似 trainer.get_rollout_inference_addresses() 的接口获取地址并传入；Proxy 启动时一次性注册，运行期间不更新。 |
| `proxy` | `ProxySpec` | 可选。Proxy frontend protocol、replica、网络暴露、session routing、trajectory 记录与保留策略。V0 固定使用 per-session random picker + sticky session。 |
| `admission` | `AdmissionSpec` | 可选。当前实验的最大并发 Task/Session、排队上限和 backpressure 策略。 |
| `default_lifecycle` | `LifecycleSpec` | 可选。Task timeout、cancel、cleanup 和 artifact retention 的默认值；TaskSpec 可以提供 task 级 override。 |

V0 不提供 `result_delivery` 启动配置项：统一使用 `SubmitTask + WaitAny long-poll`，由 Driver SDK 封装为 `TaskFuture + as_completed`。Callback 不属于 V0。

### TaskSpec （对外）

TaskSpec 描述单个 Task 的业务语义。同一 Agent Service 实例内的不同 Task 可以选择不同 AgentLoop、problem、environment 和 reward，但不能改变服务启动时已经选定的 ExecutionBackend、Proxy 部署拓扑或上游 inference replica 集合。

| 字段 | 类型 | V0 内容与边界 |
| --- | --- | --- |
| `problem` | `ProblemSpec` | 必填。prompt/messages 和相关文件。 |
| `agent` | `AgentSpec` | 必填。描述运行哪个 Agent 实现、Agent 自身参数（e.g. max_turn），以及它使用的 Proxy frontend protocol。 |
| `execution` | `ExecutionSpec` | 必填。描述如何启动 Agent artifact。V0 只定义 command 方式。 |
| `environment` | `EnvironmentSpec` | 可选。描述该 Task 的 Agent 可交互环境和环境工具；纯对话任务可以为空。 |
| `reward` | `RewardSpec` | 必填。VerifierSpec、RewardFunctionSpec。 |
| `generation` | `GenerationSpec` | 必填。sampling params、单次生成 token 限制和总 token/request budget。 |
| `lifecycle` | `TaskLifecycleSpec` | 可选。timeout、priority、cancel/cleanup override；缺省时继承 AgentServiceStartupConfig.default_lifecycle。 |

#### 一个 Search Agent 的完整 TaskSpec 示例

*Search Agent TaskSpec（V0）*
```python
task_spec = TaskSpec(
    problem=ProblemSpec(
        messages=[
            {
                "role": "system",
                "content": "Use the search tool when needed, then return a concise final answer.",
            },
            {
                "role": "user",
                "content": "Which paper introduced the Transformer architecture?",
            },
        ],
        assets=[(文件、多模态等信息...)],
        metadata={
            "dataset": "asearcher",
            "sample_id": "train-000123",
        },
    ),

    agent=AgentSpec(
        artifact="./agents/search_agent",
        config={
            "max_turns": 64,
        },
        frontend_protocol="openai_chat_completions",
    ),

    execution=ExecutionSpec(
        command=["python", "main.py"],
        working_dir=".",
    ),

    environment=EnvironmentSpec(
        variables={
            "RETRIEVAL_SERVICE_URL": "http://127.0.0.1:8001/retrieve",
            "CRAWL_SERVICE_URL": "http://127.0.0.1:8001/crawl",
        },
        setup_command=None,
        tools=[
            ToolSpec(
                name="search",
                description="Search documents relevant to a query.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results.",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
                command=["python", "tools/search.py"],
            ),
        ],
    ),

    reward=RewardSpec(
        verifier=VerifierSpec(
            kind="answer_match",
            reference_answers=["Attention Is All You Need"],
            normalization=["strip", "casefold"],
        ),
        reward_function=RewardFunctionSpec(
            kind="binary",
            correct_reward=1.0,
            incorrect_reward=0.0,
        ),
    ),

    generation=GenerationSpec(
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=4096,
        max_model_requests=64,
        max_total_tokens=131072,
    ),

    lifecycle=TaskLifecycleSpec(
        timeout_seconds=1800,
        priority=0,
    )
)
```

字段可见性与运行含义：

- `problem`、`agent`、`execution`、`environment` 经过裁剪后传给 AgentRuntime；上游 inference replica 列表仅由 Proxy 使用，`reference_answers` 等 RewardSpec 私有字段只交给 RewardEngine。
- `artifact="./agents/search_agent"` 定义 artifact 根目录；`working_dir="."` 指这个根目录，因此实际执行入口是 `./agents/search_agent/main.py`，tool executable 是 `./agents/search_agent/tools/search.py`。

#### 不得放入 TaskSpec 的 Agent Service 启动级字段

- ExecutionBackend kind、Ray address、K8s cluster/namespace、Local/Sandbox 调度配置；这些属于 AgentServiceStartupConfig.execution。
- 上游 LLM replica endpoint 列表、model alias、credential reference，以及 Proxy replica、监听端口和部署拓扑；这些属于 AgentServiceStartupConfig.inference / proxy，不得放入 TaskSpec。
- `task_id`、`session_id`：由 AgentTaskController 接收 SubmitTask 时生成，属于 Task 运行上下文，不是 TaskSpec 的业务字段。

### TaskExecutionSpec

AgentTaskController 将 AgentServiceStartupConfig 与 TaskSpec 解析为内部 TaskExecutionSpec；Proxy 则使用启动配置中的 inference replica registry 创建 sticky ProxySession。TaskExecutionSpec 不是 Driver 侧 API，也不等于原始 TaskSpec：

*AgentTaskController 解析启动配置与 TaskSpec，并最小暴露给 AgentRuntime*
```text
AgentServiceStartupConfig + TaskSpec
        ↓ resolve / validate
TaskExecutionSpec
  = Agent 可见的 problem/agent/environment 子集
  + session_id
  + Proxy endpoint/credential
  + Backend 已解析的 runtime launch config

AgentServiceStartupConfig.inference.replica_endpoints
  → Proxy 启动时注册
  → create_session(session_id) 时随机选择并保存 selected_replica
  → 不传给 AgentRuntime

RewardSpec、hidden reference、标准答案
  → 仅交给 RewardEngine，不传给 AgentRuntime
```

## 伪代码

#### Driver 侧

1. TaskRunner 从 verl 配置构造 AgentServiceStartupConfig，启动并持有当前实验的 Agent Service；服务 READY 后创建客户端 AgentExecutor，并把 RolloutAdapter 注入 trainer.fit()。

*Verl 启动并持有单实验 Agent Service*
```python
from agent_service import (
    AgentExecutor,
    AgentServiceLauncher,
    AgentServiceStartupConfig,
    InferenceSpec,
)


@ray.remote
class TaskRunner:
    def run(self, config):
        trainer = RayPPOTrainer(...)
        service_handle = None
        executor = None
        rollout_adapter = None

        try:
            # 先启动训练 Worker 和 rollout inference replicas；
            # 不再创建 Verl 原生 AgentLoopManager。
            trainer.init_workers()

            # V0：启动 Agent Service 前获取一次 replica 地址。
            # Proxy 启动后注册该固定列表。
            replica_endpoints = (
                trainer.get_rollout_inference_addresses()
            )
            startup_config = AgentServiceStartupConfig(
                execution=config.agent_service.execution,
                inference=InferenceSpec(
                    replica_endpoints=replica_endpoints,
                    model_alias=config.rollout.model_alias,
                    upstream_protocol=(
                        config.agent_service.upstream_protocol
                    ),
                    credential_ref=(
                        config.agent_service.inference_credential_ref
                    ),
                ),
                proxy=config.agent_service.proxy,
                admission=config.agent_service.admission,
                default_lifecycle=config.agent_service.default_lifecycle,
            )

            service_handle = AgentServiceLauncher.start(startup_config)
            service_handle.wait_ready()

            executor = AgentExecutor(
                service_url=service_handle.url,
                max_in_flight=config.agent_service.max_in_flight,
            )
            rollout_adapter = RolloutAdapter(executor=executor)
            trainer.fit(rollout_adapter=rollout_adapter)

        finally:
            if rollout_adapter is not None:
                rollout_adapter.cancel_in_flight()
            if executor is not None:
                executor.close()
            if service_handle is not None:
                service_handle.stop(
                    graceful_timeout_seconds=(
                        config.agent_service.shutdown_timeout_seconds
                    )
                )
```

1. RolloutAdapter是写在verl/slime侧的 一个薄适配层，不是 Agent Service 内部组件。它负责训练框架的数据（e.g. DataProto) 与 Agent Service这边的TaskSpec/TrajectoryResult 之间的格式转换，以及任务的批量提交、等待以及结果顺序恢复。

*RolloutAdapter*
```python
from agent_service import TaskId, as_completed


class RolloutAdapter:
    """在 Verl DataProto 和 Agent Service API 之间做转换。"""

    def __init__(self, executor):
        self.executor = executor
        self.in_flight: set[TaskId] = set()

    def submit_batch(self, batch, config):
        indexed_task_ids = []

        for index in range(len(batch)):
            task_spec = build_task_spec(
                item=batch[index],
                config=config,
            )
            task_id = self.executor.submit(task_spec=task_spec)
            indexed_task_ids.append((index, task_id))
            self.in_flight.add(task_id)

        return indexed_task_ids

    def wait_batch(self, indexed_task_ids, timeout: float):
        position_by_task_id = {
            task_id: index
            for index, task_id in indexed_task_ids
        }
        snapshots = [None] * len(indexed_task_ids)

        for snapshot in as_completed(
            executor=self.executor,
            task_ids=position_by_task_id.keys(),
            timeout=timeout,
        ):
            index = position_by_task_id[snapshot.task_id]
            snapshots[index] = snapshot
            self.in_flight.discard(snapshot.task_id)

        return build_verl_dataproto(snapshots)

    def cancel_in_flight(self):
        for task_id in tuple(self.in_flight):
            self.executor.cancel(task_id)
```

1. 以RayPPOTrainer为例，trainer.fit()里用RolloutAdapter往Agent Service 提交和收取任务。

*RayPPOTrainer 使用 RolloutAdapter 提交 Task*
```python
def fit(self, rollout_adapter: RolloutAdapter):
    wait_timeout = self.config.agent_service.wait_timeout_seconds

    for epoch in range(current_epoch, self.config.trainer.total_epochs):
        for batch in self.train_dataloader:
            indexed_task_ids = rollout_adapter.submit_batch(
                batch=batch,
                config=self.config,
            )
            combined_gen_output = rollout_adapter.wait_batch(
                indexed_task_ids,
                timeout=wait_timeout,
            )

            # 剩余流程沿用 RayPPOTrainer.fit()，但不再由 Trainer 计算 reward。
            self.compute_advantages(batch)
            self.compute_old_logprob(batch)
            self.update_actor_and_critic(batch)
```

*Driver 侧客户端 SDK*
```python
TaskId = NewType("TaskId", str)


class AgentExecutor:
    def submit(self, task_spec: TaskSpec) -> TaskId: ...
    #priority V0

    def get_status(self, task_id: TaskId) -> TaskSnapshot: ...
    #使用场景：一个任务hang了，查询原因，如果是pending，那可能资源给的不够多；如果是reward阶段了，那就无所谓符合预期
    #v1再做

    def cancel(self, task_id: TaskId) -> TaskSnapshot: ...
    #使用场景：样本太stale了，可以cancle任务, priority: V0

    def wait_any(
        self,
        task_ids: Iterable[TaskId],
        timeout: float,
        max_results: int,
    ) -> list[TaskSnapshot]: ...
    #priority: V0


def as_completed(
    executor: AgentExecutor,
    task_ids: Iterable[TaskId],
    timeout: float,
) -> Iterator[TaskSnapshot]:
    """反复调用 executor.wait_any，按完成顺序 yield TaskSnapshot。"""
    ...


@dataclass
class TaskSnapshot:
    task_id: TaskId
    status: TaskStatus
    created_at: datetime
    completed_at: datetime | None
    trajectory: Trajectory | None
    reward: float | None
    error: TaskError | None
```

## Phase plan

v0: no replay buffer, local ExecutionBackend only, no http server on agent service (ray rpc only but with http serializable argument passing) @Xiaole Guo

v1: ray execution backend

v2: 看neo proto的进展，用replay buffer，agent service写数据给replay buffer，只回meta给driver, trainer去buffer里拉数据

v3: 社区找人做 sandbox k8s

TODO: 把driver侧调用的api写了，跟望哥对一下。
