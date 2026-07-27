# TITO 代码实现

1. Class

init: allowed append roles

_assert_append_only: 有必要check allowed append roles吗？

用verl utils里面的apply_chat_template会不会有什么问题，参考下miles？

Incremental tokenization是否要分组，是否有必要给tool构造dummy assistant，dummy assistant为什么一定要有对应的tool call元数据NousResearch/Hermes-2-Pro-Llama-3-8BNousResearch/Hermes-3-Llama-3.1-8B看看这个跟legacy路径测试出来不一样到底是什么原因？如果不构造dummy assistant，但是有tool name,id啥的，还能输出tool tag么？

_build_assistant_message里的OpenAIFunctionCallSchema

Encode incremental user， user+user有的template会报错，所以user是不是也要构造assistant dummy为好？那么tool 和user到底要不要分开，感觉好像可以就用一个encode_incremental函数，函数里也调用同一个build_assistant_dummy，但是一个传tools,一个不传tools就行

Merge result

构造dummy的代码

Merge_token，加generation prompt，为什么要单独提取generation prompt，为什么不可以就是在apply chat template to  incremental的时候就generation_prompt = True

如果tool response是system prompt, qwen3,5会有bug

GLM的generation是不是一定会产生<observation>

1. agent loop集成

create_continuous_token看下，真的需要那么多参数吗？

tool call如果失败了，processing state如果解析出来tool response是个普通user msg怎么办？

Tool call解析失败，看下目前的tool会返回什么？然后tito是否会有问题

Tool call id Tool agent loop里如果解析出来了就一定要保留这个tool call id到messages吗？构造dummy assistant一定要有tool call id吗？看下kimi那里到底该怎么做

Verl sglang/vllm能不能直接去接sglang的openai的message返回

CTB路径真的需要维护全量messages吗？（我觉得要，因为generation promtp提取需要，但是generation prompt提取好像也不是必须需要）

verl自己的apply_chat_template是否还要保留qwen3.5 excetion 处理

lossmask和logprob，是CT来align,还是agentloop来align

Tool agent loop为什么需要完整的message呢？

qwen-next是什么？要支持一下

潜在的bug

1. toolAgentLoop里面的handle_generating_state里最开头的塞一个tool call stop token
2. Tool response啥时候会是system
3. 看下Zephyr的special token边界问题，这个应该是能说明verl 去掉sys prompt做法的缺点。其他verl 去掉sys prompt的做法的缺点，要么其实很好解决，要么miles也解决不了。
4. 如果同时call了两个一样的tool，没有tool call id的话如何区分呢？
5. GLM generation prompt是<think>，但是如果模型本身不输出think的内容呢？可以看看glm为什么full和ct没有align

TODO

1. 看测试结果；2. 到底要不要全量msg，generation prompt会有不同吗；3. Merge result；4. Tokenid level做切分；5. 为什么不可以直接对tool msg做add generation prompt = true而是要额外用full msg 来提generation prompt，看下incremental 分group到底有无必要
2. 测试trajectory加一下assistant msg里有thinking的
3. 写一下token sequence comparator
4. Pr pre commit
5. agentloop不要改传给llm的sampling param
6. Toolagentloop: c. Tool parser怎么实现的
7. 把ct的那些方法写成异步的，这样agentloopbase里是不是就不用helper function了

ContinuousTokenCore

1. 当前是否要做allowed_appended_roles
2.

测试

- [x] 看下trajectory构造

- [x] 确认下测试对assistant msg是不是做了tokenization

- [x] 看下是不是每个agentloop类型和每个trajectory都测到了

- [x] 看下测试是否有根据对应的model来加载对应的tokenizer

- [x] 实现qwen3 qwen35 GLM47

1. 跑一遍完整测试

Question to ask

1. Tool 返回的是user/system的case还没测，因为verl tool agent loop只支持role = tool
2. dpsk没有chat template 所以还没测
3. Kimi seed glm 都没有tool parser，与其自己加，不如直接拿sglang, vllm传回来的msg
4. 目前是把之前的路径也保留了，用if else enable_continuous_token来判断要用什么，要不要直接把之前的方式去掉了

High Priority

1. Gpt oss, gemma4是否有multimodal的要处理 好像legacy就没有考虑multimodel因为encode tool response是用的tokenizer而不是processor
2. 训练侧的multi modal input和推理侧真正用的是否一样
3. qwen3.5 如果只有systemprompt，CT路径要注意一下
4. Kimi VL post processing image会炸，legacy也会炸
5. deepseek
6. Verl utils里tokenzer.py hf_processor设置processor.get_rope_index，这是在绑mrope吗，verl本身好像只支持了qwen glm，那么其他vl模型该弄什么rope index呢
7. 如果把ctb改成默认，在toolagentloop和singleturnagentloop里每一轮都需要判断是否有图，然后检查当前ctb是不是是vl的？
8. Pre commit
9. 单独测下gemma4 text
10. dpsk删掉
11. TODO： mock trajectory补上reasoning_content不为空的，checker和compare里要测这种reasoning_content不为空的。
12. qwen3 vl 开thinking后，prompt后面的generation prompt和全量apply chat template后prompt和assitant output之间的那个generation prompt是不一样的，一个是\n，一个是\n\n
13. gemma4 text ctb的_tokenize_generation_prompt_delta写的有点奇怪啊
14. 除了qwen glm以外模型上gpu看停在哪里
15. gemma4的chat template 也太奇怪了，如果对全量apply chat template，会把assistant content放在了该tool call对应的tool response的后面

- [x] Template checker 看下glm 4.6v能否通过

- [ ] 把legacy删掉 那么agentloopbase里是不是甚至可以不要self.apply_chat_template_kwargs/mm_processor_kwargs了？

- [ ] **Known test gap (TODO):** the current mock trajectories never exercise a non-empty `reasoning_content` on assistant turns — every mocked assistant message has empty/absent reasoning. A follow-up should add a case with non-empty `reasoning_content` to verify CT rendering / boundary handling stays token-equivalent when reasoning is present. (text和vl都有这个gap）

- [x] 对拍 看下glm 4.6 和legacy的对比 以及和全量的对比

- [x] 看下gemma4的text对拍（✅）和vl 对拍（✅），post processing（✅）

- [ ] 上gpu看下gemma4到底生成停止在什么token呢？

- [x] 检查gpt oss对拍

- [x] 看下如果是非agent loop 会不会用ctb？（答案：不会）

- [x] 看下识别processor那里，glmv4.6是不是可以直接加上（✅）确认下verl image transformer版本里是否有glm46v（✅）

- [x] Ctb create 传processor+mm_processor_kwargs Unit test加一下kwargs的

- [x] 比较四个dump，加mm之前的，merge conflict之前的，merge conflict但是保留legacy的，删掉legacy的
  - [x] Merge conflict 之后 vs 之前 （text+vl都有，不包含gemma4）
  - [x] Merge conflict 之后 VS 加vl之前（不包含gemma4）
  - [x] 删掉legacy （不包含gemma4）

- [ ] Dpsk text+vl， kimivl, minimaxVL ct支持上

- [x] mm和text unit test全过

|  | 解析CTB是否正确 | Assistant output构造正确 | 是否真的有展开image pad | CT vs Legacy | CT vs Full | 多模态encode正确性 |
| --- | --- | --- | --- | --- | --- | --- |
| **qwen 2.5** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Qwen 3** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **glm** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **mimo** | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| kimi | ❌ | ✅ | ✅ | ✅ | ✅ | 炸了 |
| minimax | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| nemotron | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 | 不支持 |
| gemma4 | ✅ |  | ✅ |  |  |  |
