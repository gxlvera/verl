# [RFC] 在 Verl 中支持 Multi-turn Continuous Token

## Summary

本RFC提议在 Verl 的multi-turn rollout 中 (AgentLoop / AgentGateway）引入一套通用的 Continuous Token 机制。Verl 目前已经是 token-in-token-out：模型输入是 token ids，模型输出也会以 token ids 进入 trajectory。但Continuous Token关注的问题不仅仅是“是否 token-in-token-out”，而是 multi-turn 场景下如何可靠地维护多轮token：上一轮模型输出、环境反馈、下一轮 assistant generation prompt 之间的边界如何正确处理，以及如何兼容不同模型的特化行为，以及如何实现可复用、可配置、可校验。同时，相关逻辑不应该写死在AgentGateway/AgentLoop里，而是通过建立一个抽象层来进行封装，且允许模型特化行为。实现上，P1 先采用进程内 Python util，由 `AgentGateway` / `AgentLoop` 直接调用`ContinuousTokenBuilder`统一的  API；如果后续 profiling 证明 tokenization 成为延迟或吞吐瓶颈，再考虑 把tokenization单独做成服务。本 RFC 覆盖两条 agentic rollout 路径的集成：

- Existing / legacy Agent Loop：`verl/experimental/agent_loop` 及当前 multi-turn rollout request schema
- New Agent Framework / Gateway：`verl/agent/framework` 与 `verl/agent/gateway`，即 session-based、OpenAI-compatible gateway path。

## Motivation

Verl 目前已经具备基础的 token-in-token-out 形态：首轮 prompt 会被 tokenization 成 input ids，模型生成的 token ids 会直接写入 trajectory，环境 token 会放入 response 侧并用 `response_mask=0` 标记。但是在多轮 agent 场景里，一方面，continuous token 的拼接、边界处理仍然不够完善，在某些模型上会出错；另一方面，这些逻辑不应该直接写死在AgentLoop/AgentGateway，而是应该做成一个抽象层。

目前的实现主要体现在三类问题上：

- 多轮拼接逻辑写死在 session 层或具体 `AgentLoop` 里，不同路径各自维护增量 tokenization、mask/logprob 对齐和 message history 更新。这部分逻辑应该抽象成可复用的 builder / merge API，供 Gateway、legacy Agent Loop、rollout request schema 共享。
- 不同模型族的 chat template 不同，continuous token 在 assistant stop token、下一轮 role boundary、tool response 渲染、trailing newline 等边界上的处理方式也会不同。框架需要提供模型族 adapter，允许 Qwen、GLM 等已知模型实现特化 merge 行为，而不是把这些规则散落在调用侧。并且，当前的实现没有对边界进行处理，从而会导致rollout拼接出来的数据不符合apply_chat_template的预期行为，这会影响训练的效果。
- 多轮拼接后的 token stream 缺少合理的结构化校验机制。仅做完整 canonical render 的文本 diff 容易把 assistant runtime token 与模板 canonicalization 的预期差异误报；需要引入 comparator，按角色和 special token boundary 区分 hard mismatch 与 assistant content soft mismatch。

**verl当前实现的具体问题及案例可参考：[Agentic Multi-turn Continuous Token Design Notes](https://bytedance.us.larkoffice.com/docx/L9JedwS23olB3NxnkxAukLaWsxe)，**重点看“一”里面的2和3，里面有一些verl目前会出bug的example

## Goals

- 引入 `ContinuousTokenBuilder` 抽象，封装 multi-turn continuous token 的核心逻辑，包括 runtime prefix 复用、环境消息增量 tokenization、boundary merge 以及 mask/logprob 对齐所需的 merge metadata。
- 支持 model-family 级别的特化行为，boundary merge、encode non-assistant msg 特化，例如缺失 trailing newline、stop token 与下一条消息 start token 重叠、tool response boundary 依赖前序 assistant tool call 等。
- 在 finalize/debug 阶段提供结构化 comparator：hard mismatch 只覆盖 special token、role boundary、non-assistant content；assistant content mismatch 仅作为 metric 或 debug 信息。
- 为  AgentGateway 与Agent Loop 共用同一套实现。
- 默认保持向后兼容：未开启 Continuous Token 时现有逻辑不变。

## Proposed Design

### Shared Continuous Token module

新增共享模块：

```text
verl/utils/chat_template/continuous_token.py
verl/utils/chat_template/token_seq_comparator.py
verl/utils/test_utils/chat_template_verify.py
```

核心抽象为 `ContinuousTokenBuilder`。负责复用上一轮 runtime token prefix，只对新追加的非 assistant 消息做增量 tokenization，并在 prefix / delta 交界处应用模型族特化的 boundary merge。P0 仅覆盖 text-only path，因此首轮 prompt 构建直接使用 `render_messages(..., tokenize=True)`；结构化返回值留到 P2 multimodal path 真正需要 processor outputs 时再引入。

```python
class ContinuousTokenBuilder:

    def __init__(
        self,
        tokenizer: Any,
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs or {}

    def create_comparator(self) -> TokenSeqComparator:
        return TokenSeqComparator(
            tokenizer=self.tokenizer,
            assistant_start_str=self.assistant_start_str,
            trim_trailing_ids=self.trailing_token_ids or None,
        )

    def append_assistant_tokens(
    self,
    runtime_token_ids: list[int],
    assistant_token_ids: list[int],
) -> list[int]:
    """Append rollout-produced assistant tokens to the runtime prefix. Default behavior is concat.
    """
    return list(runtime_token_ids) + list(assistant_token_ids)

    def render_messages(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str | list[int]:
        """Render via the model chat template.

        Used by initial prompt rendering, dummy-context suffix diff,
        comparator / verifier, and fallback full render.
        """
        return apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **self.chat_template_kwargs,
        )

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _tokenize_rendered_suffix(
        self,
        base_messages: list[dict],
        appended_messages: list[dict],
        *,
        tools: list[dict] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render base and base+append, then encode only the rendered suffix."""
        text_without = self.render_messages(
            base_messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
        )
        text_with = self.render_messages(
            base_messages + appended_messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
        assert text_with.startswith(text_without)
        return self._encode_text(text_with[len(text_without):])

    def _dummy_system() -> dict[str, Any]:
        """Minimal stable system context used only for suffix-diff rendering."""
        return {"role": "system", "content": "dummy system"}

    def _dummy_assistant_with_tool_calls(tool_messages: list[dict]) -> dict[str, Any]:
        """Build a synthetic assistant that opens the same tool-response boundary.

        Tool response rendering commonly depends on the previous assistant
        message's tool_calls, especially call ids and function names. The dummy
        assistant should preserve those structural fields, but its natural-language
        content / reasoning is irrelevant because it is removed by suffix diff.
        """
        return {
            "role": "assistant",
            "content": "",
            # Some reasoning templates expect this key to exist. The exact text is
            # intentionally dummy because only the suffix after this assistant is used.
            "reasoning_content": " ",
            "tool_calls": [
                {
                    "id": tool_msg.get("tool_call_id") or f"call0000{i}",
                    "type": "function",
                    "function": {
                        "name": tool_msg.get("name") or "dummy_func",
                        # Arguments are not used to render the following tool
                        # response boundary; keep them structurally valid.
                        "arguments": {},
                    },
                }
                for i, tool_msg in enumerate(tool_messages)
            ],
        }

    def _tokenize_tool_segment(
        self,
        tool_messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ) -> list[int]:
        """Encode a contiguous tool-response run using synthetic context."""
        return self._tokenize_rendered_suffix(
            [dummy_system(), _dummy_assistant_with_tool_calls(tool_messages)],
            tool_messages,
            tools=tools,
        )

    def _tokenize_user_and_system_segment(
        self,
        message: dict,
        *,
        tools: list[dict] | None = None,
    ) -> list[int]:
        """Encode a single user/system append with the smallest stable context."""
        return self._tokenize_rendered_suffix(
            [_dummy_system()],
            [message],
            tools=tools,
        )

    def _split_appended_segments(
    self,
    appended_messages: list[dict],
) -> list[list[dict]]:
        """Group appended messages into template-stable segments.

        P0 behavior:
        - contiguous tool responses are encoded as one segment
        - each user/system message is encoded as one segment
        - assistant append is rejected; assistant tokens must come from rollout
        """
        ...

    def _encode_environment_delta(
        self,
        old_messages: list[dict],
        new_messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ) -> list[int]:
        """Encode non-assistant messages appended after old_messages.

        It validates message-level append-only history, encodes appended
        tool/user/system segments, and appends the next assistant generation
        prompt exactly once.
        """
        assert_append_only_with_allowed_roles(
            old_messages,
            new_messages,
            self.allowed_append_roles,
        )

        appended = new_messages[len(old_messages):]
        delta_ids: list[int] = []
        for segment in self._split_appended_segments(appended):
            role = segment[0]["role"]
            if role == "tool":
                delta_ids.extend(self._tokenize_tool_segment(segment, tools=tools))
            elif role in {"user", "system"}:
                delta_ids.extend(
                    self._tokenize_user_and_system_segment(segment[0], tools=tools)
                )
            else:
                raise ValueError(f"unsupported appended role: {role}")

        # The assistant opener depends on the full post-append history.
        delta_ids.extend(
            self._tokenize_rendered_suffix(
                new_messages,
                [],
                tools=tools,
                add_generation_prompt=True,
            )
        )
        return delta_ids

    # Backward-compatible alias if implementation wants to mirror Miles naming.
    tokenize_additional_non_assistant = encode_environment_delta

    def merge_tokens(
        self,
        old_messages: list[dict],
        new_messages: list[dict],
        pretokenized_token_ids: list[int],
        *,
        tools: list[dict] | None = None,
    ) -> MergeResult:
        """Merge runtime prefix with encoded environment delta.

        Default behavior is concat. Model-family subclasses override this to
        insert or trim boundary tokens while reporting those mutations in
        MergeResult so masks/logprobs can be updated consistently.
        """
        delta_ids = self._encode_environment_delta(
            old_messages,
            new_messages,
            tools=tools,
        )
        return MergeResult(
            token_ids=list(pretokenized_token_ids) + delta_ids,
            delta_ids=delta_ids,
        )
```

`MergeResult` 显式表达交界处修改，而不是只返回拼好的 token ids：

```python
@dataclass
class MergeResult:
    token_ids: list[int]
    prefix_trim_count: int = 0
    inserted_boundary_ids: list[int] = field(default_factory=list)
    delta_ids: list[int] = field(default_factory=list)
```

这样 gateway / agent loop 可以同步更新 `response_ids`、`response_mask`、`response_logprobs`，避免某些模型（比如 GLM4.7）在 prompt token stream 中已经移除了 stop token，但训练 trajectory 的 mask/logprob 仍保留旧 token。

### Model-family adapters

提供 registry：

```python
default
qwen3
qwen35
glm47
```

model family adapter 一般只overwrite merge_tokens()，其他都用default builder里的方法。

- Qwen 类：模型可能停在 `<|im_end|>`，而模板下一轮需要 `<|im_end|>\n`，merge 时插入 newline 并将该 newline 标成 `response_mask=0`。
- GLM 类：某些 boundary token 既可能是 assistant stop token，也可能是下一条消息 start token，merge 时需要从 runtime prefix 尾部移除一个 ambiguous boundary token，并同步 trim mask/logprob。

配置上用户可配置以下项：

```yaml
actor_rollout_ref:
  rollout:
    multi_turn:
      continuous_token:
        enable: True
        model_family: auto   # auto | default | qwen3 | qwen35 ｜ glm47
        allowed_append_roles: ["tool","user"] # ["tool"] | ["tool","user"]
        hard_mismatch_policy: warn   # warn | error | disable
```

model_family如果为auto，框架通过 checkpoint推断出model_family，然后registry根据model family自动解析需要的ContinuousTokenBuilder；如果用户显式传了这些配置，则以用户配置为准。

### Comparator

新增 `TokenSeqComparator`，按 special token boundary 切 segment：

- `special_token_count`：segment 数量或 special/content 结构不同，hard mismatch。
- `special_token_type`：special token 类型不同，hard mismatch。
- `non_assistant_text`：user/system/tool 等非 assistant 内容不同，hard mismatch。
- `assistant_text`：assistant 内容不同，soft mismatch。原因是 assistant token 来自模型真实输出，可能包含 reasoning content、额外自然语言或未 canonicalize 的内容；Continuous Token 不应在训练侧重写它。Comparator 用于 finalize debug、metric、CI verifier，不用于改变 token stream。

### Verifier Tool

提供一个一键验证工具，用于在接入主链路前验证某个模型族的 chat template 是否满足 Continuous Token 的拼接假设：

```bash
python scripts/verify_continuous_token_chat_template.py \
  --model Qwen/Qwen3-... \
  --continuous-token-builder qwen3 \
  --allowed-append-roles tool user
```

- 通过 registry 创建真实的 `ContinuousTokenBuilder` 与 tokenizer，而不是只检查 raw jinja template。
- 使用 mock trajectories 覆盖 single tool、多 tool、parallel tool、user retry、system injection、reasoning / thinking content 等典型多轮场景。
- 对每个 case 同时计算 full canonical render 与 continuous-token merge 结果，验证 `decode(merged_ids)` 是否与 reference render 对齐，并输出首个 mismatch 的上下文。
- 支持 `--allowed-append-roles`、model-family builder、chat template override、template kwargs 等参数，便于在新增模型或调整 template 时作为本地检查与 CI verifier 使用。
- 失败时应区分 template 本身不满足 append-only/continuous-token 假设，还是 builder 的 boundary merge 特化缺失，方便定位是改 template、改 adapter，还是收窄支持矩阵。

## 与现有agentic rollout的集成

### AgentGateway Integration

1. Gateway 初始化时根据 config 创建 `ContinuousTokenBuilder`，替代当前 `_system_prompt = initialize_system_prompt(...)` 与 `_encode_incremental(...)` 的增量逻辑。Gateway 只保存 builder 引用，不直接持有 tokenization helper。
2. 首轮请求走 builder 的 full render 高层入口：

```python
rendered = continuous_token_builder.render_messages(
    messages,
    tools=tools,
    add_generation_prompt=True,
)
prompt_ids = rendered.token_ids
```

1. 模型生成后通过 builder 追加 assistant output，避免调用侧写死 `prompt_ids + assistant_output_ids`：

```python
runtime_ids = continuous_token_builder.append_assistant_tokens(runtime_ids, assistant_output_ids)
```

1. 后续请求命中 message/tool prefix 时，用当前 active trajectory 的 runtime token stream 做 environment merge：

```python
merge = continuous_token_builder.merge_tokens(
    old_messages=session.message_history,
    new_messages=messages,
    runtime_token_ids=runtime_ids,
    tools=tools,
)
```

1. 根据 `MergeResult` 更新 buffer：

- 如果 `prefix_trim_count > 0`，从 `response_ids`、`response_mask`、`response_logprobs` 尾部同步裁剪。
- `inserted_boundary_ids` 与 `delta_ids` 追加到 `response_ids`。
- 这些环境/boundary token 的 `response_mask` 为 0，logprob 填 0.0。
- `generation_context_ids = merge.token_ids` 传给 rollout backend。

1. decode 出 assistant message 后更新 `session.message_history = messages + [assistant_msg]`。
2. `_materialize_active_trajectory` 或 `finalize_session` 阶段运行 comparator，把 mismatch summary 写入 `Trajectory.extra_fields`，并按 `hard_mismatch_policy` 决定 warn 或 raise。

### AgentLoop Integration

1. `AgentLoopBase.__init__` 中根据 config 创建 `ContinuousTokenBuilder`，并把 tokenizer、processor、`apply_chat_template_kwargs`、model family、allowed append roles 等 tokenization 相关配置都交给 builder。AgentLoop 只保存 builder 引用，不再提供 `AgentLoopBase.apply_chat_template(...)` 这类 tokenization helper。
2. 首轮 prompt 构建走 builder 的 full render 入口。AgentLoop 不直接调用 `apply_chat_template(...)`：

```python
agent_data.prompt_ids = continuous_token_builder.render_messages(
    messages=agent_data.messages,
    tools=active_tool_schemas,
    add_generation_prompt=True,
)
```

1. 模型生成阶段维持 token-in-token-out 语义，但 assistant output append 也通过 builder 完成：

```python
agent_data.prompt_ids = continuous_token_builder.append_assistant_tokens(agent_data.prompt_ids, assistant_token_ids)
```

1. 解析 tool call 后，需要把本轮 assistant message 追加到 `agent_data.messages`。这样后续 tool response merge 时，builder 能看到前序 assistant `tool_calls`，并据此构造正确的 tool-response boundary：

```python
agent_data.messages.append(assistant_message)
agent_data.tool_calls = tool_calls
```

1. tool response / user retry / system injection 等 environment message 追加时，`ToolAgentLoop` 不再对新增消息做 `apply_chat_template(..., remove_system_prompt=True)`，也不再在 AgentLoop 层特殊判断 GPT-OSS。它只把旧 history、新 history 和当前 runtime token prefix 交给 builder：

```python
old_messages = agent_data.messages
new_messages = old_messages + add_messages

merge = continuous_token_builder.merge_tokens(
    old_messages=old_messages,
    new_messages=new_messages,
    runtime_token_ids=agent_data.prompt_ids,
    tools=active_tool_schemas,
)
```

1. `ContinuousTokenBuilder.merge_tokens(...)` 内部负责 environment delta tokenization、dummy assistant construction、assistant generation prompt、model-specific boundary merge，并返回 `MergeResult`。AgentLoop 只根据 `MergeResult` 更新 state：

- `agent_data.prompt_ids = merge.token_ids`
- `prefix_trim_count > 0` 时同步裁剪 `response_mask` / `response_logprobs`
- `inserted_boundary_ids + delta_ids` 作为 environment/boundary token 追加，`response_mask=0`，`response_logprobs=0.0`
- `agent_data.messages = new_messages`

1. `SingleTurnAgentLoop` 不需要多轮 merge，但首轮 prompt 也应通过 `continuous_token_builder.render_prompt(...)` 构建，保证 AgentLoop 层没有直接 tokenization 入口。
2. finalize/debug 阶段可调用 builder/comparator 做结构化校验，把 hard/soft mismatch 统计写入 output extra fields，并按 `hard_mismatch_policy` 决定 warn 或 raise。

`verl/workers/rollout/schemas.py` 的 `AsyncRolloutRequest` 里也存在 turn-by-turn 拼接逻辑，后续可以迁移到同一套 `ContinuousTokenBuilder`，用新的 comparator 替代当前的纯文本 diff sanity check。

## Compatibility and Migration

- 默认 `continuous_token.enable=false`，现有行为不变。
- 第一阶段仅支持text；processor/multimodal path 可以保留原逻辑并打印 unsupported/fallback 日志。
- `use_inference_chat_template` 与 Continuous Token 语义不同：前者允许用 inference template 重建 prompt；Continuous Token 强调 runtime token stream。启用 Continuous Token 时建议忽略或禁止 `use_inference_chat_template=True`(?)
- `tokenization_sanity_check_mode` 可以保留，但 Continuous Token comparator 应成为新的默认检查方式。

## Test Plan

测试分两层：先验证 `ContinuousTokenBuilder` 自身的 continuous-token 假设，再验证它与verl传统拼接路径的行为差异。

1. Builder-level comparator tests`ContinuousTokenBuilder` 需要先通过我们自己的 `TokenSeqComparator`。测试输入使用 mock multi-turn trajectories，覆盖 single tool、多 tool、parallel tool、user retry、system injection、reasoning / thinking content、assistant stop token、trailing newline、tool response boundary 等典型场景。每个 case 同时计算：然后用 `TokenSeqComparator` 比较 reference 与 candidate：
  - reference：对完整 message history 做 full canonical render。
  - candidate：用 `ContinuousTokenBuilder` 逐轮执行 `render_messages`、`append_assistant_tokens`、`merge_tokens` 得到 continuous token stream。
  - special token / role boundary / non-assistant content mismatch 视为 hard mismatch。
  - assistant content mismatch 视为 soft mismatch，只记录 metric/debug 信息。
  - hard mismatch 在 CI 中失败；soft mismatch 默认不失败，但需要输出 summary。
2. AgentLoop A/B token-id tests在 verl  AgentLoop 上构造同一批 rollout case，分别跑两条 token 拼接路径：测试直接比较两边最终的 token id 序列，以及 `response_mask` / `response_logprobs` 长度对齐关系。对已经确认 baseline 有问题的模型族，可以允许 baseline 与 continuous 不完全一致，但必须满足：
  - baseline：现有 AgentLoop 的 token id 拼接结果。
  - continuous：使用 `ContinuousTokenBuilder` 的 `append_assistant_tokens(...)` 与 `merge_tokens(...)` 得到的 token id 拼接结果。
  - continuous 结果通过 `TokenSeqComparator` hard check。
  - 差异位置能归因到模型族 boundary merge，例如插入 newline、裁剪 ambiguous stop token、tool response boundary 修复。
  - `MergeResult` 中的 `prefix_trim_count`、`inserted_boundary_ids`、`delta_ids` 能解释 mask/logprob 的变化。
3. Tokenizer / model-family coverage

测试矩阵需要覆盖多类 tokenizer 和 chat template，而不是只测一个 template。至少覆盖：

- Qwen / Qwen3 / Qwen3.5
- GLM / GLM4.x
- DeepSeek
- Mistral / Mixtral
- Kimi
- Seed
- GPT-OSS / Harmony-style tool response

每个模型族至少覆盖 text-only multi-turn tool calling；multimodal processor path 可以作为 P2 单独扩展。

## Limitations

- 对复杂 multimodal template 的增量 tokenization 需要后续单独验证。

## Implementation Plan

- P0 核心数据结构实现
- P1 与现有AgentLoop/AgentGateway集成+跑通example
- P2 支持多模态

注：

P0先将 Continuous Token 实现为进程内 Python util，而不是独立 service 或 Ray actor。每次 agent request 通常只需要对新增的非 assistant messages 做 delta tokenization，单次开销预期较小；拆成独立服务反而会引入 Ray RPC、序列化/反序列化和跨进程传输成本。若使用 stateless `ContinuousTokenActor`，每次还需要传入 `old_messages`、`new_messages`、`tools` 以及 runtime/prompt token ids，payload 也可能不小。因此 P1 的重点是先封装统一的 builder / merge API，让 `AgentGateway` 和 `AgentLoop` 直接 in-process 调用，在保持低开销的同时维持模块解耦。后续补充 `continuous_token_render_ms`、`continuous_token_encode_ms`、`continuous_token_merge_ms`、`gateway_wait_ms` 等 profile metrics；如果数据证明 tokenization 成为 critical-path latency 或高并发吞吐瓶颈，再考虑 offload 到 executor或者把tokenization这块儿做成一个单独的service。

####
