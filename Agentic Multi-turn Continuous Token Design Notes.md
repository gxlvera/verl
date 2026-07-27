# Agentic Multi-turn Continuous Token Design Notes

## TL; DR

Token-in-token-out (TITO) 的基本概念是：训练侧应该使用推理侧真实产生和看到的 token ids，而不是事后重新 tokenization。verl 目前已经具备最基础的 token-in-token-out 思路：它会保留上一轮累计的 token ids，并正确对齐loss mask。但在 agentic 多轮 rollout 中，涉及到多轮token的拼接。很多开源模型的 chat template 并不是简单 append-only 的；**如果 incremental token ids 的提取和边界拼接处理不严谨，会拼出错误的多轮 prompt。**如果只应对某个模型的tokenzier，那么其实可以通过特定方式来确保正确性，但我们希望能设计一套方案来泛化到不同的tokenzier上，并且尽量少地去做model-specific hardcode。可泛化的多轮TITO主要有四个容易踩坑的点：

1. 多轮 Continuous Token不能重新 retokenize 当前 full messages，必须保留之前轮累计的真实 token ids。
2. incremental non-assistant token ids 必须从合适的 synthetic context 中提取，不能简单 encode incremental messages。
3. incremental ids 拼回旧 token ids 时，部分模型需要显式的边界处理。
4. 构造完整 trajectory 后，需要 comparator 检查结构错误；可以容许 assistant 内容与 full chat template 的 canonical render有偏差，但不可以容许special token边界的偏差。

其中，第 1 点是 TITO 的核心不变量，应该由 trajectory/session 层固定；第 2、3 点是模型相关的实现细节，可以通过模型特化策略重写；第 4 点是最终的结构校验机制。

本文会讨论以上四个点verl和公司内部的实现情况，以及说明如何通过`ContinuousTokenBuilder`抽象实现“**核心不变量pipeline+模型部分特化方法**”来实现正确的多轮tito。

## 一、多轮 Continuous Token要注意的问题

### 1. 不可以对 full messages 重新 retokenization

多轮 rollout 里，上一轮模型看到的是：

```text
prompt_token_ids + model_output_token_ids
```

下一轮继续生成时，应该直接复用这串真实 token ids。如果重新把 `full_messages` 过一遍 `apply_chat_template + encode`，会有两类问题。

1. BPE 边界变化的问题。模型可能在推理时分两步生成了两个 token：

```text
output_ids = [A, B]
decode([A, B]) = " international"
```

但如果把字符串 `" international"` 放回 message 里重新 encode，BPE 可能把它 merge 成一个 token：

```text
encode(" international") = [C]
```

这样训练侧的 token 序列就不是推理侧真实产生的 token 序列。

1. chat template 位置相关渲染。某些 template 会根据 assistant message 是否是历史消息、是否包含 `</think>`、是否处在最后一个 user 之后，来插入或删除 token。重新 render full messages 会把推理时真实存在的 prompt token 改掉。

#### Seed 内部的padding 方法

Seed 内部的 padding 方法可以应对以上的第一个问题，但是无法应对第二个问题。因为它依赖一个更强的假设：把旧 assistant content 替换成 padding 后，chat template 在 padding 前后的结构不变。这个假设对 Seed 这类较规则的 tokenizer 可能成立，但无法推广到 GLM4.7等会根据 assistant 内容或位置改写 thinking/generation prompt 的模板。Seed 内部的方法本质上仍然是重新 apply chat template，但它把旧 assistant content 换成等长 padding，试图避免 BPE 边界变化：

```python
# Turn 1: 推理侧真实生成
output_ids = [1234, 3456, 4567]

pad_str = "?"*len(output_ids) # assuming that we have verified "?" will never be BPE merged

# Turn 2: 重新 apply template 前，把旧 assistant 替换为等长 pad
full_msgs_padded = [
    system_msg,
    user_msg,
    {"role": "assistant", "content": pad_str},  # len(encode(pad_str)) == len(output_ids)
    next_user_msg,
]

padded_ids = encode(apply_chat_template(full_msgs_padded))

# 然后把 pad span 替换回 output_ids
restored_ids = padded_ids[:pad_start] + output_ids + padded_ids[pad_end:]
```

这个方法能解决 BPE 边界变化的问题：assistant 的真实输出 token ids 不再由 `decode -> encode` 得到，而是直接塞回 pad span。但这个方法隐含要求：

```text
把 assistant content 替换成 pad 后，pad 前后的 template token 不变。
```

这个要求对 general chat template 不成立。

#### 反例：DeepSeek-R1-Distill-Qwen-1.5B {folded="true"}

DeepSeek-R1-Distill-Qwen-1.5B 的 Turn 0 generation prompt 是：

```text
<｜Assistant｜><think>\n
```

对应 token 末尾是：

```python
[151645, 151648, 198]
# 151645 = <｜Assistant｜>
# 151648 = <think>
# 198    = "\n"

```

模型真实生成的 output ids 是：

```python
output_text = "I should add one and one.\n</think>\n1+1 = 2"
output_ids = [40, 1265, 912, 825, 323, 825, 624, 151649, 198, 16, 10, 16, 284, 220, 17]
```

真实历史是：

```text
<｜Assistant｜><think>
I should add one and one.
</think>
1+1 = 2
```

如果用 padding 重新 apply chat template，DeepSeek 历史 assistant 分支会把 `<think>\n` 去掉（因为判断"000000000000000"里不含thinking，所以chat template会把generation prompt里的\n去掉，这是他chat_template自己的逻辑），padded render 变成：

```text
<｜Assistant｜>000000000000000<｜end▁of▁sentence｜><｜User｜>...
```

或者如果按照原始 output start index 去 restore，会得到类似：

```text
<｜Assistant｜>00I should add one and one.
</think>
1+1 = 2...
```

因为 padded 序列里缺了推理时真实存在的 `<think>\n` 两个 prompt token，后续位置已经不同构。这个问题不是 BPE，而是 generation prompt token 被 completed-message template 吃掉了。

#### 反例：GLM4.7 {folded="true"}

GLM4.7 默认生成 prompt 是：

```text
<|assistant|><think>
```

但历史 assistant message 分支如果没有 `reasoning_content`，或者 padding content 里没有 `</think>`，会渲染：

```text
<|assistant|></think>{content}
```

所以如果旧 assistant 被替换成 padding，pad 前的 marker 会从真实 runtime 的：

```text
<|assistant|><think>
```

变成：

```text
<|assistant|></think>
```

这个 token 在 pad span 前面，restore pad 覆盖不到。

#### 结论

我们必须保留之前轮累计的真实 token ids：

```python
accumulated_token_ids = previous_prompt_ids + previous_completion_ids
```

下一轮只能追加新的 incremental token ids，不能对整个 `full_messages` 重新 encode。在这一点上，verl 当前逻辑是对的：它保留旧 trajectory buffer，下一轮将 incremental ids append 到 `response_ids`。例如 gateway 里 prefix 命中后会 copy active buffer，然后追加 incremental ids，见 `verl/verl/agent/gateway/gateway.py:524-545`。问题在于后面两件事：**incremental token ids 怎么提取**，以及 **怎么 merge 到旧 token ids 上**。

### 2. 怎么获得 incremental token ids

message-level incremental 很容易得到：

```python
incremental_messages = new_messages[len(old_messages):]
```

难点是：如何把这些 incremental messages encode 成“纯净的、应该追加到旧 token ids 后面”的 token ids。如果直接：

```python
encode(apply_chat_template(incremental_messages, add_generation_prompt=True))
```

很多 template 会额外生成 system prompt、tool schema、role prelude 等。这些内容本来已经在最初 prompt 或旧 token ids 中出现过，不应该再次拼进 trajectory。

#### verl 当前做法

verl 已经发现了这个问题，所以 `_encode_incremental` 会单独 encode incremental messages，然后删掉 `self._system_prompt`：

```python
ids = apply_chat_template(incremental_messages, add_generation_prompt=True)
return ids[len(self._system_prompt):]
```

源码位置：

- `verl/verl/agent/gateway/gateway.py:412-454`
- 传统 ToolAgentLoop 也有同样的 `remove_system_prompt=True`，见 `verl/verl/experimental/agent_loop/agent_loop.py:243-309`但这个方法有两个问题。

#### 问题 1：system prompt 长度提取本身不可靠

Two bad cases: https://github.com/verl-project/verl/issues/6500 https://github.com/verl-project/verl/issues/6501

verl 的 system prompt 提取在 `verl/verl/utils/chat_template.py:13-32`：

```python
token1 = apply_chat_template([user_empty], add_generation_prompt=False)
token2 = apply_chat_template([user_empty, user_empty], add_generation_prompt=False)
system_prompt = token1[:-(len(token2) - len(token1))]
```

这个方式假设第二个 user 只是追加了一段固定长度 suffix，所以 len(token2) - len(token1) 就是第一条 user turn 的长度。

但有些 tokenizer 的 turn delimiter / eos / thinking marker 会随消息位置变化。比如一个 message 在“最后一条”时和在“历史消息”时渲染不同，`len(token2)-len(token1)` 就不再是稳定的 user turn 长度。这样提取出的 `system_prompt` 长度会错，后续 incremental ids 的起点也会错。所以我们更像倾向于用构造anchor message的方式来提取纯净的incremental token。但是如果随意构造anchor message会引发问题B。

#### 问题 2：tool response 通常依赖前面的 assistant tool call

tool response 不是普通 user 文本。很多 chat template 需要看到：

```python
{"role": "assistant", "tool_calls": [...]}
{"role": "tool", "content": "..."}
```

才能正确渲染 tool-response boundary。只对 `tool` message 单独 apply chat template，或者只用 anchor system，很容易直接报错或者得到错误边界。比如某些模板需要根据前一个 assistant tool call 来决定如何渲染。所以构造anchor message也需要特定的方式，并且允许不同模型重写方法。

##### 反例1: MiniMax

如果没有dummy assistant chat template会直接报错

##### 反例2: Nemotron

如果没有dummy assistant 会少渲染 <|im_start|>user

##### 反例3: Llama-3.1-8B-Instruct

如果我们为了提取 tool response 的 incremental ids，只构造 dummy system：

```python
base_messages = [
    {"role": "system", "content": "dummy system"},
]

appended_messages = [
    {
        "role": "tool",
        "tool_call_id": "call12345",
        "name": "get_weather",
        "content": '{"temperature":"20C"}',
    }
]

text_without = tokenizer.apply_chat_template(
    base_messages,
    tools=tools,
    tokenize=False,
    add_generation_prompt=False,
)
```

这个 tokenizer 会直接报错：

```text
TemplateError: Cannot put tools in the first user message when there's no first user message!
```

原因是这个 template 在有 `tools` 的时候，会试图把 tool schema 放进第一条 user message；但 synthetic context 里只有 system，没有 user，也没有前置 assistant tool call，所以这个上下文对它来说是不合法的。如果构造 dummy assistant：

```python
base_messages = [
    {"role": "system", "content": "dummy system"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call12345",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "SF"},
                },
            }
        ],
    },
]

full_messages = base_messages + appended_messages

```

就可以正常 render，并且 suffix diff 能切出干净的 tool result：

```text
<|start_header_id|>ipython<|end_header_id|>

"{\"temperature\":\"20C\"}"<|eot_id|>

```

这个例子说明：dummy assistant 的作用不是“随便垫一条 assistant”，而是在 synthetic context 里恢复真实 rollout 的前置状态：**assistant 刚刚发起了对应 tool call，现在才轮到 tool response**。这里还有一个小细节：构造 tool-response synthetic context 时不要顺手加 dummy user。比如 Qwen3 会根据“assistant 是否在最后一个真实 user 之后”“assistant 是否是最后一条消息”来决定是否插入或清理 `<think>...</think>`。如果 base 写成 `[_DUMMY_SYSTEM, _DUMMY_USER, dummy_assistant]`，`text_without` 里 dummy assistant 可能因为位于最后一个 user 之后而被插入空 thinking block；但 `text_with = base + tool_response` 后 assistant 又不再是最后一条，thinking block 被删掉，导致 `text_with.startswith(text_without)` 失败。这个设计的重点是：**不要求完整 conversation 的 chat template append-only （append-only是指，对msgs1+msgs2 apply chat template后的前缀完全等于只对msgs1 apply chat template），只要求 synthetic context 下的append-only即可，稳定切出当前 role 的 suffix即可。**

### 3. incremental ids 如何正确 merge 回 accumulated ids

拿到 incremental token ids 后，verl的做法是简单的：

```python
accumulated_ids + incremental_ids
```

这个做法是不严谨的。因为模型 stop token 和 chat template 的下一个 turn boundary 之间，经常不是一一对应。

#### 反例1 Qwen3：生成停在 `<|im_end|>`，template 需要 `<|im_end|>\n` {folded="true"}

Qwen3 chat template 的每个 message 结尾通常是：

```text
<|im_end|>\n
```

但模型生成时 stop 在：

```text
<|im_end|>
```

不会生成后面的 `\n`。如果下一轮直接 append incremental ids，就少一个 newline。

#### 反例2 GLM4.7：stop token 和下一条消息 BOS 共用

GLM4.7 里 `<|user|>` 和 `<|observation|>` 既可能是 assistant 的 stop token，也可能是下一条消息的 start token。例如 tool call 成功时：

```text
runtime prefix ends with: <|observation|>
incremental tool response starts with: <|observation|><tool_response>...
```

如果直接拼：

```text
...<|observation|><|observation|><tool_response>...
```

就出现重复 `<|observation|>`。tool call 失败或没有走 tool 时，模型可能已经停在：

```text
<|observation|>
```

但下一轮环境追加的是 user retry：

```text
<|user|>tool call fails, please retry
```

直接拼会变成：

```text
...<|observation|><|user|>tool call fails...
```

语义也错了，因为 `<|observation|>` 是模型为了 tool result 预期生成的 stop/boundary，但真实下一轮不是 observation。注意，经验表明special token的错误是语义层面的，比qwen3的\n问题要严重，可能会有较大训崩的风险。

#### 结论

综合以上三点，我们需要一个 `ContinuousTokenBuilder` 抽象：

```python
class ContinuousTokenBuilder:
    def encode_environment_delta(...):
        ...

    def merge(prev_messages, next_messages, runtime_token_ids, tools=None):
        delta = self.encode_environment_delta(...)
        return [*runtime_token_ids, *delta]
```

不同模型可以通过继承 ContinuousTokenBuilder 并重写实例方法来定制化增量提取和边界拼接。针对一些常用模型，我们可以写好正确的 model-specific continuous token builder 以供用户直接使用。

### 4. trajectory 构造后需要 comparator

完整 trajectory 拼好后，需要检查：

```text
non-assistant 内容是否正确
special token 数量/类型是否正确
message boundary 是否正确
```

但不能简单要求：

```python
accumulated_ids == encode(apply_chat_template(full_messages))
```

因为 full template canonical render 可能会修改历史 assistant 内容。比如 Qwen3 / DeepSeek / GLM 可能会清掉之前轮的 thinking：

```text
runtime actual:
<think>
previous reasoning
</think>
final answer

canonical full render:
final answer
```

这种差异应该允许，因为 TITO 的目标是保留真实 runtime token prefix，而不是让历史 assistant 被 template 事后 canonicalize。比较逻辑：

1. 先按 special token 切 segment。
2. 如果 segment 数量或 special/content pattern 不同，报 `special_token_count`。
3. 如果 special token 类型不同，报 `special_token_type`。
4. 如果普通 content 不同，根据它是否属于 assistant segment，分类为 `assistant_text` 或 `non_assistant_text`。`assistant_text` 是 soft mismatch：允许 assistant 犯错，允许 runtime assistant token 和 canonical template 不同。hard mismatch 是：

```text
special_token_count
special_token_type
non_assistant_text
```

这些说明 role boundary、tool response、user/system/env 内容等被 TITO 拼坏了。verl 当前没有这样的 comparator。

## 二、正确的做法和设计

### 数据结构

```python
# 一个 RolloutTokenState 对应一次 agentic rollout 的一条 sample trajectory。
class RolloutTokenState:
    rendered_messages: list[dict]
    runtime_token_ids: list[int]
    records: list[RequestResponseRecord]
```

### Continuous token builder 抽象

```python
class ContinuousTokenBuilder:
    append_only_roles = {"tool"}

    def render_chat(self, messages, *, add_generation_prompt, tools=None, tokenize=False):
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **chat_template_kwargs,
        )

    def encode_render_delta(self, anchor_messages, delta_messages, *, tools=None, add_generation_prompt=False):
        before = self.render_chat(anchor_messages, add_generation_prompt=False, tools=tools)
        after = self.render_chat(
            anchor_messages + delta_messages,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        if not after.startswith(before):
            raise ValueError("synthetic context is not append-only for this segment")
        return encode(after[len(before):])

    def encode_tool_observations(self, tool_messages, tools=None):
        assistant_stub = make_tool_call_stub(tool_messages)
        return self.encode_render_delta(
            [SYNTHETIC_SYSTEM, assistant_stub],
            tool_messages,
            tools=tools,
        )

    def encode_plain_turn(self, message, tools=None):
        return self.encode_render_delta(
            [SYNTHETIC_SYSTEM],
            [message],
            tools=tools,
        )

    def encode_environment_delta(self, prev_messages, next_messages, tools=None):
        require_message_prefix(prev_messages, next_messages, append_only_roles=self.append_only_roles)
        new_tail = next_messages[len(prev_messages):]

        pieces = []
        for role, segment in group_consecutive_roles(new_tail):
            if role == "tool":
                pieces.extend(self.encode_tool_observations(segment, tools=tools))
            elif role in {"user", "system"}:
                pieces.extend(self.encode_plain_turn(segment[0], tools=tools))

        pieces.extend(self.encode_render_delta(
            next_messages,
            [],
            tools=tools,
            add_generation_prompt=True,
        ))
        return pieces

    def merge(self, prev_messages, next_messages, runtime_token_ids, tools=None):
        delta = self.encode_environment_delta(prev_messages, next_messages, tools=tools)
        return [*runtime_token_ids, *delta]

```

### 模型特化 merge

```python
class Qwen3ContinuousTokenBuilder(ContinuousTokenBuilder):
    def merge(...):
        delta = self.encode_environment_delta(...)
        prefix = list(runtime_token_ids)
        if prefix and prefix[-1] == im_end_id:
            prefix.append(newline_id)
        return [*prefix, *delta]
```

```python
class GLM47ContinuousTokenBuilder(ContinuousTokenBuilder):
    def merge(...):
        delta = self.encode_environment_delta(...)
        prefix = list(runtime_token_ids)
        if prefix and prefix[-1] in {user_id, observation_id}:
            prefix = prefix[:-1]
        return [*prefix, *delta]

```

### Runtime 链路

```python
def build_prompt_tokens(request_messages, tools):
    if state.runtime_token_ids is None:
        return continuous_builder.render_chat(
            request_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

    return continuous_builder.merge(
        prev_messages=state.rendered_messages,
        next_messages=request_messages,
        runtime_token_ids=state.runtime_token_ids,
        tools=tools,
    )
```

```python
def update_after_generation(request_messages, prompt_token_ids, completion_token_ids, assistant_message):
    # 这里保存的是 runtime 真实 token stream，不是 canonical full render。
    state.runtime_token_ids = prompt_token_ids + completion_token_ids
    state.rendered_messages = request_messages + [assistant_message]
```

### Finalize / debug comparator

```python
def inspect_runtime_render_gap(state):
    canonical_ids = continuous_builder.render_chat(
        state.rendered_messages,
        add_generation_prompt=False,
        tokenize=True,
    )
    return token_sequence_checker.compare(
        expected_ids=canonical_ids,
        actual_ids=state.runtime_token_ids,
    )

```

Comparator 只把下面几类作为 hard error：

```text
special_token_count
special_token_type
non_assistant_text
```

`assistant_text` 只记录 metric，不作为 hard error。原因是 assistant 内容来自模型真实输出，它可以包含错误格式、未 canonicalize 的 thinking、额外自然语言、甚至不符合模板预期的文本；这些不应该被 TITO 事后改写。

## 三、对 verl 当前逻辑的总结

verl 当前已经满足“基础 token-in-token-out”：

- 初始 prompt 全量 `apply_chat_template`。
- 模型输出 token ids 直接 append。
- tool/user 环境增量 token ids mask 为 0。
- 后续 generation 用累计 token ids。

但它还缺少更为 robust的multi-turn TITO 相关组件：

1. 他针对模型特化性的逻辑是通过类似`if Qwen3`这样hardcode来进行的，没有进行抽象的设计
2. incremental ids 不是通过 dummy context suffix diff 提取，而是 standalone encode 后 strip system prompt。
3. `initialize_system_prompt` 的长度推断对位置相关 template 不稳。
4. tool response 没有构造 dummy assistant/tool_calls 以保留 tool boundary shape。
5. finalize 后没有 token sequence comparator 来区分 assistant soft mismatch 和真正的 structure/non-assistant hard mismatch。
