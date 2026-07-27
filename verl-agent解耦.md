# verl/agent解耦

## **TODO**

1. Walk through slime 和slime.agent的耦合式example
2. 想一下如果要把slime.agent弄成独立服务，应该加胶水哪些代码，这些代码应该作为example里的用户代码呢，还是说放到slime.agent比较合适，类似于这些胶水代码把slime.agent服务化后返回给example里的用户代码一个slime.agent service url，然后slime就可以直接调用了。cc：“lime.agent 零改动，新增约 400–600 行（服务端 wrapper + 原位 HTTP 客户端），大半是从 generate.py 搬运。依据：数据面早已是 HTTP，`Sample` 自带 `to_dict/from_dict`，唯一要 RPC 化的是 `open/finish/drop_session` 三个本地调用。”
3. slime里的coding agent例子是真的e2e能训的吗，跑的啥harness呢
4. 看下slime和polar怎么交互
5. 看下slime和slime.agent是如何交互的，adapter service咋做的，从slime load data到 到把prompt发给agent，起sandbox这一整个call stack
6. verl是做成slime那种，slime可以本地调用slime.agent，二者跑同一个进程吗，还是说verl直接做成polar那种，verl用http调verl-agent里 任何东西
7. 看下verl要如何来做这个adapterservice
8. 看下verl agent 相关的，看下写法到底跟开源当时的generate sequence有啥不同，看下跟现在的verl的agent的写法是否一样了
9. 如果要用verl接我的agentservice跑例子，该用verl的什么trainer，如何起这个service呢？
10. 看下wang哥的verl agent interface doc
11. 看下uni agent里的ray object 调用能否跨Ray 集群，uni agent gateway内部有没有ray的东西
12. Trainer 给agent service发prompt 如果用ray rpc可以跨ray集群吗？
13. 看下slime和polar 目前到底符不符合wang哥的设计
14. 再读下wang 哥文档的代码
15. 看下为什么不从verl agentloop.run那里接submit to agent service？看下slime接agentloopmanager那些能接进去么，接verl agentloop呢？

几大框架解耦度对比

|  | 代码 | 运行时 | 交互协议 | 数据契约 |
| --- | --- | --- | --- | --- |
| Uni-agent<br>vs Verl | ❌ | 目前是跑在同一个ray cluster❌ | 传的是python对象/ray actor句柄，要求必须同一个进程/ray集群 |  |
| Slime.agent<br>Vs Slime | ✅ | 目前的例子是跑在同一个进程❌ 但是用户代码可以改成独立部署？ | Url str (e.g. Sgl server url, tool parser name) |  |
| Polar vs Slime | ✅ | ✅ | Url str |  |

## 要确定的事情

1. verl-agent里如果对于其他包的依赖，比如如果想用FunctionCallParser那是不是要import，那是不是就得装sglang的包，那么是不是就得连带装比如torch啥的，单独clone sglang改pythonpath好像也行不通，因为加载sglang某些模块还是需要比如torch。问下slime.agent那边的人如何独立部署slime.agent。slime的人说，这确实是个问题，按照目前的写法无法绕开对sgl的依赖，除非像slime那样patch sgl 直接从sgl同时拿tokenid和message
2. slime.agent单独部署的机器上要单独安装tokenizer吗，如何确定这个tokenzier是我训练侧想用的那个呢
3. 看下uni-agent refactor后的代码，是否还有import verl agentloop这些？
4. uni-agent起gateway的对象是什么,agentframework吗？可以把agentframework类比slime那个adapterservice吗，也是用户代码吗，那用户可以overide agentframework把gateway独立部署吗？
5. slime-bridge是只跑一次吗，还是每次rollout的时候都跑
6. 7.8在食堂问望哥，管理session的脚本由谁来跑，怎么跑，他说放到比如katapod里通过entrypoint注入啥的跑？
7. 望哥说session 创建不用在katapod还是sandbox创建之前？
8. verl不用transfer queue那用什么？但是replay buffer咋写的
9. slime里的openai/anthropic特有的文件是做啥的，slime有没有polar没有的东西，为什么slime群里有人说要把slime的anthropic的东西给miles miles才能跑cc呢？

针对agentservice文档

1. cc说“另有一个**真实的模糊点**值得你顺便问清：范围边界的"我们交付"列表只写了 Proxy、Agent Service、Driver SDK、operator 接口清单——**binary 侧的 AgentClient 不在交付列表里**，但 API 草图又花了一整节写它。这是 doc 自身的不一致，问一句"AgentClient 算不算我们 v0 交付"很正当。
2. “可选 `PROXY_BASE_URL`” 为什么proxy_base_url是可选的啊
3. 为什么uni-agent, slime都给anthropic和openai各做了一层adapter呢？
4. v1 separate async trainer从tq里拿的是数据还是meta？kvbatchmeta是啥？
5. 望哥问为什么要有reward engine，这是为了解决什么问题？以及中间阶段的reward，和final reward的计算有没有什么区别？
6. 如果把所有replica地址都发给proxy，看下proxy怎么来pick replica，最好是重复verl本身的LLMServerClient的逻辑
7. 如果同一个task的多次attempt都交给driver，agent service侧会为其创建不同task id，那么session id是不是没有必要存在了

**知识性问题**

一些问题

1. 问下tiannuo 跑起来coding的例子没有
2. 在一个ray actor上起一个http服务是什么意思？那这个http服务、占用的端口、和ray actor（一个进程）是什么关系呢？
3. Ray Task和Ray actor的区别？
4. Ray supervisor到底是什么意思？看 ray doc里的driver到底是我们说的verl的driver（taskrunner ray actor），还是那个init ray的普通python 进程
5. 为什么verl让taskrunner这一个ray actor创建了其他所有ray actor？这样不会有single point of failure吗？

跟Changyi meet留下的问题

1. 问下wang 哥 proxy到底是多个task共享还是每个task一个（sidecar形式）如果是sidecar，为什么changyi说sidecar一定是另外一个sandbox呢，我的proxy不能就在host的一个单独的进程上吗？changyi说，只要sidecar跟agent runtime/binary生命周期不共存亡（agent runtime和proxy不放同一个kata），就没事，这样agent runtime/binary的kata被杀死了，sidecar proxy还存在，里面就还保留有收集到的轨迹 （所以这反驳了文档里写的共用proxy的好处（即便 kata 被秒杀也完整(proxy 在 host)）。如果不用k8s，就本地进程，也有task sidecar概念吗？
2. [Cross-Platform Agent Service](https://bytedance.larkoffice.com/docx/TLicwHl4iiPjuak3adxcmYL4nmg)到底和polar有啥区别
3. changyi说现在的数据都是要先给verl，然后verl再给trainer（megatron/fsdp），agent service的设计是指传handle给verl，verl把handle传给trainer，然后trainer根据handle去replay buffer拿真正的数据
4. changyi说taskservice就是agentservice里对每个task的描述的东西，kata启动时为什么只带少量信息，然后去task service里拿task的信息呢，这不会多此一举吗？这么做有什么好处吗？
5. Changyi 说operator就是sandbox，我觉得不是吧，我觉得是管理sandbox的东西吧

要了解的：

1. 如果是k8s，sidecar proxy一定是跑在跟agentbinary一个kata pod里吗？可以跑host上一个进程吗？如果是local，sidecar proxy跑哪里？
2. operator到底是什么，polar slime.agent有无对标。
  1. operator是一个python对象吗？ operator有哪些方法，哪些是per batch
3. （generate.py+slime.agent）封装成一个服务，是不是就是polar了
4. miles和uni agent的sticky session在哪里做的呢？verl的GlobalRequestLoadBalancer是怎么做sticky session的？
5. 一个实验里，可以不同的任务用不同的task manager实现吗？比如有的任务用本地线程跑agentloop，有的任务用sandbox跑个agent loop。答案：不可以，一个实验都是一种形式的agnetloop（比如都是local或者ray），但是可以不同task用不同agent loop
6. **设计图里reward计算在哪里，看下slime miles agent reward计算放哪里的**

7.9 日跟望哥下午画设计图

1. 问题
  1. ✅如何让proxy知道推理引擎replica的地址？望哥说,推理引擎起后上报给proxy的地址，不用走driver（望哥说虽然推理引擎是driver起的，但是推理引擎可以直接把自己的地址报给proxy）。但是我觉得这个url很轻，给driver也可以，如果每个replica自己上报，那我还得改verl里llm replica相关的代码。如果走driver, 那我只在agentloopmanager submit task那里的代码把replica url加到taskSpec即可。
  - **📌 cc 整理：backend 地址上报的两种候选方案**
    - **共用前置——零 patch 取址**：bridge 经 `llm_client._load_balancer`（⚠ 私有属性，启动时 assert 兜底 + 提 upstream PR 转正）调 `get_all_servers()` 直接拿到全部 replica 地址（"ip:port" 字符串列表，verl llm_server.py 自带此 getter）。无需 replica 自报、无需改 verl 任何代码。
    - **方案一：地址列表随 TaskSpec 声明 + proxy 哈希粘性**
      - bridge 把 servers=[全部 replica URL] 放进 BackendSpec，随每个 task 声明；agent service 收到后同名 upsert（幂等自愈：service 重启后下一个 submit 自动补齐注册表）。
      - proxy 选台：hash((task_id, attempt)) % N。同一 task 永远同一台（粘性保 prefix cache）；失败则 attempt+1 换台重试；任务量大时统计均衡。
      - 零新组件、零 verl 改动。代价：统计均衡而非精确最少在途，且与 verl 内部流量是两本账。
      -
        ```python
        # bridge create 时（零 patch 取址）
        lb = llm_client._load_balancer          # 私有属性，assert 兜底
        addrs = await lb.get_all_servers.remote()          # ["10.x.x.7:30801", ...]
        backend = BackendSpec(name=f"verl-{exp}", servers=[f"http://{a}" for a in addrs])
        # 每次提交随身带：TaskSpec(..., backend=backend)；服务端 registry 同名 upsert

        # proxy 侧：哈希粘性 + 失败换台
        def pick(servers, task_id, attempt=0):
            return servers[hash((task_id, attempt)) % len(servers)]
        # ConnectError 时 attempt+1 重试即换台
        ```
    - **方案二：verl 的 LB actor 包一层 HTTP 转发，agent service 只认一个 URL**
      - bridge spawn 一个小 ray actor（把 LB handle 作为构造参数传入——handle 集群内可传，零 patch；转发进程独立于 driver，数据面不过 driver 进程），actor 内跑 HTTP 转发：每请求 acquire（粘性 + 全局最少在途）→ 转发到 server_id（其值就是 "ip:port"）→ finally release。
      - BackendSpec 只带这一个转发器 URL；proxy 无脑打它，请求头带 X-Session-Id=task_id 做粘性键。
      - 独家优势：全部 agent 流量经 acquire/release 记账，与 verl 内部流量（teacher/reward 等）共用同一本负载账。代价：数据面 +1 跳、约 100 行自维护代码、转发器随训练 job 生灭。
      -
        ```python
        # bridge：spawn 转发 actor，LB handle 当构造参数递进去
        router = LbRouterActor.options(num_cpus=1).remote(lb_handle=lb)
        backend = BackendSpec(name=..., servers=[ray.get(router.get_url.remote())])  # 只有 1 个 URL

        # LbRouterActor 内的 /generate handler：
        sid = request.headers["X-Session-Id"]              # 粘性键 = task_id
        server_id, _ = await lb.acquire_server.remote(request_id=sid)  # sticky + 最少在途
        try:
            r = await client.post(f"http://{server_id}/generate", content=body)  # 只搬字节不解析
        finally:
            lb.release_server.remote(server_id=server_id)

        # proxy 侧：打唯一 URL，带 X-Session-Id 头
        ```
    - **对比与建议**：v0 用方案一起步；BackendSpec.servers 是列表这一契约两案通用（8 个 URL ↔ 1 个 URL），切换时 proxy 零改动。观测到副本负载偏斜、或 verl 内部流量与 agent 流量显著共享副本时，升级方案二。
    - **⚠ 已排除的做法**：bridge 提交时读 LB actor 在途快照来逐 task 选址——agent 流量不经过 acquire/release，那本账看不见 agent 负载（rollout 期间的主要流量全在账外），且突发批量提交会因快照不变产生羊群效应压死单台。原则：负载账本只有站在流量的必经之路上才是真的；不在路上，要么自己记账（bridge 记每台活跃 task 数），要么问副本本人（sglang 负载指标端点）。
  1. **✅**望哥说agent service如果直接把trajectory数据（很大，比如1g）回传给driver，然后driver再给trainer，这样的不好的点就是driver就得等很久才收到数据（真的需要吗）然后就不能干别的事情了，但是如果是agentservice里每个task thread直接把数据传给buffer，只把handle给driver，那driver就只需要等收到轻量的data？然后driver把handle拿给trainer，然后trainer再自己去buffer里拿大数据。而且 driver是单点的，每个大数据都经过一下单点的不太合理，但是如果把replay buffer也做成单点的那就没太大区别了。望哥说**v0不要replay buffer，回传所有数据即可为什么不能数据走driver？**）不能被数据面的耗时间的任务（比如序列化、反序列化）卡住。
    1. driver是单点的（由于他的性质，他就得是单点的，不方便水平扩容），大量数据打一个单点的，不合理容易打爆；而且单点的带宽有限，容易成为带宽瓶颈。
    2. driver是控制器，他的任务更多是控制面的，要求延时低反应快，比如（
    - 掉队者治理：超时判定、发 `Cancel`——事件循环卡 30 秒 = 僵尸任务多跑 30 秒；
    - 失败重提交：task FAILED 要及时补提，否则拖长本步尾延迟；
    - 回调/心跳的 ACK（callback 模式下 driver 自己跑着一个 HTTP server，卡住 = 对面超时重试风暴）；
    - 提交下一波（one-step-off/流水线形态的吞吐来源）；
    - metrics 上报、checkpoint 记账、响应健康检查。
  1. local跑thread还是docker这个是用户决定的吗？用户怎么传递他的决定呢？
  3. 望哥说现在verl这种把agentloopmanager没有独立跑在一个ray actor上的情况而是跑在driver上的情况，那如果一个agenttask挂了，那么agentloopmanager会感知所以driver也会感知？但是把agentloopmanager（就是我们的operator）做出来，driver就不感知了，那就是operator自己去重启agentloop。
  4. 望哥说driver 给agentservice发东西，不一定要http。
  5. 我用户定义agent loop的文件（一个python脚本）放在哪里呢，望哥说只要放在能被访问到的地方就行，比如thread里跑那就放在sys path能访问的地方，如果放docker里跑就mount进去（怎么mount）
  6. 望哥说task info里的env不是sandbox的环境配置，而是要在哪里跑，比如是subcoroutine吗还是咋的？看看agent service doc里的task spec
  7. Agent service可以单独起吗？就是不由driver拉起。像proRL那样，启动脚本单独起。
2. 跟望哥已经confirm的点
  1. Agent task必须先把数据都放到replay buffer里面，等数据都传完了，才能返回handle给driver。
  2. Claude code / codex不能在 local thread上跑，但是可以试试本地docker里起container跑，虽然目前大家一般不会这么跑，但是应该可以试试。
  3. ray版本的也许可以搬verl现在的agentloopmanager那些
  4. 给reward，一般有两步，首先是判断这个task对不对，这个是会依赖环境的，比如依赖sandbox里目前的agent写的代码和其他文件状态；然后根据这个task对不对给reward，这一步有可能也要依赖环境（比如如果agent的代码没通过测试，那么是什么报错，是哪方面的问题，可能对应了不同的reward）。所以希望把最终reward的计算也拿给agent service做，反正agent service要返回一个最终的reward给训练。如何判断这个task答得对不对，是dataset里面有的判断逻辑。到那时根据task答得对不对，计算reward是算法定义的。需要把算法定义爹reward计算逻辑传给agentservice。
  5. 如何起agentservice，望哥说slime的话提交ray job的时候，有一个ray supervisor，然后把agent service作为一个角色加进去。看下verl这边怎么起？
  6. 虽然agent service部署不解耦，但是代码解耦，并且agent service模块化了，成为了一个自己的闭环，这样别的框架比如slime也能轻松接入。

7.12  codex comment，要确认的点

1. 【high】Task spec到底包含哪些东西
2. 【high】ServiceManager, TaskManager, RuntimeExecutor职责、关系
3. rollout权重变了要不要给agentservice上报version，一个trajectory跨version谁来管
4. 什么是 discriminated union，用kind作为discriminator又是什么意思？
5. http和gRPC有啥区别？跟ray有啥关系
6. 看下verl agentloop运行完后是在哪里做的teacher inference
7. AgentClient是什么，我的设计是否需要
8. Proxy sidecar的cpu开销在哪些地方？
9. proxy.finalize_session()里面干了啥
10. taskmanager两种方式区别（感觉是不是，如果类继承，无法避免用户把不该override的也override了）
11. 为什么codex说要用proxyprovider

Now 7.13:

1. Task spec finalize （v0 mvp✅），✅改成taskattempt
2. AgentServiceClient怎么收到返回，是像wang哥文档那样吗？还是走回调？✅ 用waitany
3. ✅（已更新到rfc）删掉execution handle/ref，用session id
4. reward到底是reward engine弄，还是execution backend弄

```text
1. 是否有必须独立的安全边界？
   是 → 考虑独立 Evaluation Service

2. 是否有多个非 Agent-Service 上游？
   是 → 考虑共享 Evaluation Service

3. Evaluation 是否需要脱离 AgentTask 长期存在？
   是 → 考虑独立 Job Service

4. 是否只是资源类型或扩缩容不同？
   是 → 先拆独立 Worker Pool，不必做完整 Engine

5. SGLang、ExecutionBackend、Task Store 是否已覆盖调度能力？
   是 → 不重复建设 Reward Engine

6. 剩余需求能否由 Agent Service + Evaluator 解决？
   能 → 不拆
```

1. driver侧load了data后，如何调agentexecutor submit task ✅
  1. AgentServiceRolloutProvider存在的必要（可以确认的有：数据契约转换，validate 复用 ，兼容旧路径）。
  2. dataproto和task spec怎么转？⭕️
2. ⭕️ Create experiment的时候要传一个lifecycle，timeout有哪几种类型，跟agentloop意外终止有什么区别，怎么处理呢？

Now 7.15 （写driver侧启动、call agent service代码）

1. 让codex fable分别设计
2. 两种cancle的场景，看下slime和verl的all abort是咋搞的；abort 接口要不要有？
3. 看下verl slime miles有没有不是update weight 导致的abort and resume？目前verl agent loop 怎么做abort and resume的？是sgl侧返回abort然后agentloop返回吗？还是agentloop收到abort后不断轮循，然后sgl resume后agentloop照常进行？正在算reward的时候abort会怎么样？verl什么时候会abort agentloop怎么abort，get status verl slime 挂了那agent service不也挂了吗
4. Ray Object store, r3 expert

7.20

1. Uni agent 里swe bench， task config, swebenchtask是什么？针对那种agentloop本身就跑在sandbox里的，还能用swebenchtask抽象，task.run_sandbox(), task.run_agent()吗？问下ding yuyang为什么cc没有用swebenchtask.run() - > cc agent.run()这条链路。
2. ✅一个进程里跑1024个coroutine不会太重吗 答案：不会
3. uni-agent是不是还没弄seed sandbox？
4. Claude code runner到底是哪一个？为什么不就用react那个task runner，以及cc sandbox能否用seed sandbox，还是只能用Akernel + sidecar+gateway tunnel
5. Ray worker和ray actor的关系？
6. Uni agent cc 例子为什么要用ray task而不是inline_async？以及

```python
ray_task 可以把这些控制逻辑分散、隔离到普通 Ray worker 进程中，避免全部压在唯一的 AgentFrameworkWorker event loop 里。
代价是：
Ray task 调度开销。
需要较多 Ray worker 进程。
每个长期等待 Sandbox 的 task 默认仍占用一个 logical CPU slot。
runner 每次都要重新实例化。
```

什么叫“每个长期等待 Sandbox 的 task 默认仍占用一个 logical CPU slot。”

1. Local execution backend，能直接用启动一个本地进程跑一个黑盒agent吗，比如本地进程跑claude code？
2. task抽象，其实不应该叫create_sandbox，应该叫create_env
3. Uni agent refactor后，Ray task感觉有问题：1. 如果512 task同时跑，是不是得需要512进程；2. Ray task是同步的

7.21

1. 看下codex对于cc黑盒实现难度评估；
2. 看下uni agent新版链路分析 codex回答；
3. 看下gpt对于e2e例子的搜索结果
4. 看下灰盒会不会好做一点；
5. 用uni agent + seed sandbox复现cc inference

7.24

1. Prefix match到底是token level还是，message + tool schema level？
2. 共享前缀的loss mask到底如何设置，为什么会造成共享前缀的梯度权重变大的问题？

**Code: implementation后续要check的问题**

1. ‼️agent service wait any轮询会不会阻塞agent service，毕竟ray是单线程的。每一个我创建的Ray actor，都注意下是否得是**async actor**？
2. Fully async 怎么用rollout adapter来取tasksnapshot，不用wait_batch的话（看下wang哥文档里的demo
3. Task spec怎么定义？是要有一些固定字段吗，还是直接把dataproto的字段透传？回答：目前是有一些固定字段，这样agent service才能读取相关信息，然后把dataproto其他字段作为extra fields传入。目前跟legacy已对齐，至少不会比legacy少传，但是固定字段和extra field可能有重复。看下uni agent怎么做的
4. tokenizer/model path传给proxy，如何保证proxy这边能访问到这个文件？回头测一下proxy加载出来的tokenizer/processor跟verl legacy是一样的，用例子测试一下
5. Client sdk是不是也不一定要用as completed，直接循环wait any，收集好一个batch后直接全部返回？
6. codex说：

用户重写 `AgentRuntime` 也合理，不过建议通过可序列化 spec 和 registry 创建：

```text
AgentRuntimeSpec(
    kind="sandbox_command",
    artifact=...,
    config=...,
)
```

不要把 Python class/object 直接塞进 `TaskRunSpec`，否则 K8s、跨语言 worker、版本管理都会比较难做。

1. agent和taskrunner的cpu资源声明写在哪里
  1. Task runner里面，run_agent定义得跟uni agent太像了
2. Proxy server的rollout back怎么处理（先看看gateway和miles）
3. 注意一下claude code的max token，想下如果截断了会咋样
4. Verl init tokenzier = false可能导致sglang读不到eos
5. 如果一个trajectory fail了，那么是不是这一组都无法训了？trainer怎么知道这个trajectory是不是fail了呢？

Design思考：

1. 目前的设计是，统一task runner，通过重写比如SWEEnv和SWEEval实现task的特化；比起uni agent的那种task就直接写成可继承的(SWETask)，然后重写整个run方法，这两种有什么区别呢？
2. **为什么要让executionbackend包含taskrunner，而不只是agentloop呢？包含task runner的意义是什么？**
3. 明确的点：local, task runner只能跑协程上，至于agent跑协程还是进程，这是agent里自己定义的；ray， task runner只能跑ray worker上的协程上，至于agent是协程还是进程，这是agent自己定义的。问下这种设计，k8s的是否好接入，sandbox其实不能算是一种mode？

Timeline

week-6: 把rfc伪代码定下来

week-5: ppt✅ + 写agent service

Todo:

1. 看codex设计的抽象，把controller->executionbackend->task runner ->agentruntime写了
2. 把对外接口服务侧的写了
3. 把dataproto - > task spec搞了
4. 把proxy写了
5. 跑a search的例子+swebench react agent 例子
6. Uniagent e2e训出来，我的inference代码跑通，modal sandbox调通

week-4 -2（三周）： agentservice代码+local跑通

Week - 1: opsd, delta weight sync细化profile，present等

---

## Appendix：Client SDK 的 TaskFuture 与 as_completed

> **契约约定：** 服务端提供 `WaitAny`；`WaitAny` 返回 `0..max_results` 个完整 `TaskResult`。SDK 的 `as_completed()` 用一组 `task_id` 调用 `WaitAny`，`TaskFuture.result()` 用单个 `task_id` 调用同一个 `WaitAny`。远端 Task 的失败由 `TaskResult.status` 表达；网络、协议和本地等待超时才作为 SDK 异常抛出。

### 共享契约

```python
# agent_service/contracts/task.py

from dataclasses import dataclass
from typing import Any


TERMINAL_STATUSES = {
    "SUCCEEDED",
    "TRUNCATED",
    "FAILED",
    "ABORTED",
    "CANCELLED",
}


@dataclass
class TaskResult:
    task_id: str
    status: str
    trajectory: Any | None = None
    reward: float | None = None
    error: dict | None = None
    metadata: dict | None = None


@dataclass
class WaitAnyRequest:
    experiment_id: str
    task_ids: list[str]
    wait_timeout_seconds: float = 30
    max_results: int = 32


@dataclass
class WaitAnyResponse:
    # 0..max_results 个；空列表表示本次 long-poll 窗口超时。
    completed_tasks: list[TaskResult]
```

服务端对外接口为 `POST /v1/tasks:waitAny`。请求体和响应体分别使用 `WaitAnyRequest` 与 `WaitAnyResponse`，客户端与服务端共享同一份 schema。

### 服务端 WaitAny

```python
@router.post("/v1/tasks:waitAny")
async def wait_any_route(
    request: WaitAnyRequest,
) -> WaitAnyResponse:
    return await controller.wait_any(request)


class AgentTaskController:
    async def wait_any(
        self,
        request: WaitAnyRequest,
    ) -> WaitAnyResponse:
        self.validate_task_ownership(
            experiment_id=request.experiment_id,
            task_ids=request.task_ids,
        )

        completed = await self.task_store.find_terminal_results(
            experiment_id=request.experiment_id,
            task_ids=request.task_ids,
            limit=request.max_results,
        )
        if completed:
            return WaitAnyResponse(completed_tasks=completed)

        subscription = self.completion_notifier.subscribe(
            experiment_id=request.experiment_id,
            task_ids=request.task_ids,
        )

        try:
            # 订阅后重查一次，避免查询与订阅之间的完成事件丢失。
            completed = await self.task_store.find_terminal_results(
                experiment_id=request.experiment_id,
                task_ids=request.task_ids,
                limit=request.max_results,
            )
            if completed:
                return WaitAnyResponse(completed_tasks=completed)

            try:
                await subscription.wait(
                    timeout=request.wait_timeout_seconds,
                )
            except TimeoutError:
                pass

            completed = await self.task_store.find_terminal_results(
                experiment_id=request.experiment_id,
                task_ids=request.task_ids,
                limit=request.max_results,
            )
            return WaitAnyResponse(completed_tasks=completed)

        finally:
            subscription.close()
```

`TaskStore` 是结果的事实来源，`CompletionNotifier` 只负责唤醒 long-poll。等待窗口超时返回空列表，不会取消远端 Task。

### 底层 Client

```python
# agent_service/sdk/client.py

class AgentServiceClient:
    def wait_any(
        self,
        request: WaitAnyRequest,
    ) -> WaitAnyResponse:
        response = self.http.post(
            f"{self.service_url}/v1/tasks:waitAny",
            json=serialize(request),
            # HTTP read timeout 略大于服务端 long-poll 窗口。
            read_timeout=request.wait_timeout_seconds + 5,
        )
        response.raise_for_status()
        return deserialize(
            response.json(),
            WaitAnyResponse,
        )
```

### TaskFuture

```python
# agent_service/sdk/future.py

import time


class TaskFuture:
    """远端 Agent Task 在 Driver 侧的惰性代理。"""

    def __init__(
        self,
        client: AgentServiceClient,
        experiment_id: str,
        task_id: str,
    ):
        self.client = client
        self.experiment_id = experiment_id
        self.task_id = task_id
        self._result: TaskResult | None = None

    @property
    def is_resolved(self) -> bool:
        return self._result is not None

    def _resolve_from_remote(
        self,
        result: TaskResult,
    ):
        assert result.task_id == self.task_id
        assert result.status in TERMINAL_STATUSES
        self._result = result

    def result(
        self,
        timeout: float | None = None,
    ) -> TaskResult:
        # as_completed() 已解析过时，直接读取本地缓存。
        if self._result is not None:
            return self._result

        deadline = (
            None
            if timeout is None
            else time.monotonic() + timeout
        )

        while self._result is None:
            if deadline is None:
                wait_window = 30
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Waiting for {self.task_id} timed out"
                    )
                wait_window = min(30, remaining)

            # 单 Task 等待通过单元素 WaitAny 实现。
            response = self.client.wait_any(
                WaitAnyRequest(
                    experiment_id=self.experiment_id,
                    task_ids=[self.task_id],
                    wait_timeout_seconds=wait_window,
                    max_results=1,
                )
            )

            if not response.completed_tasks:
                continue

            self._resolve_from_remote(
                response.completed_tasks[0]
            )

        return self._result

    def cancel(self):
        # 同步 Client SDK：收到取消请求确认后返回，
        # 不保证 AgentRuntime 已经完成退出和清理。
        return self.client.cancel_task(self.task_id)
```

`TaskFuture.result(timeout=...)` 中的 timeout 是 Driver 的总等待上限，不是 Agent execution timeout；本地等待超时不会自动取消远端 Task。

### SDK as_completed

```python
# agent_service/sdk/completion.py

import time


def as_completed(
    futures: list[TaskFuture],
    timeout: float | None = None,
):
    if not futures:
        return

    # V0：同一批 Future 必须来自同一个 Client 和 Experiment。
    client = futures[0].client
    experiment_id = futures[0].experiment_id

    for future in futures:
        assert future.client is client
        assert future.experiment_id == experiment_id

    pending = {
        future.task_id: future
        for future in futures
    }

    deadline = (
        None
        if timeout is None
        else time.monotonic() + timeout
    )

    while pending:
        # 先交付已经被其他调用解析过的 Future。
        resolved_task_ids = [
            task_id
            for task_id, future in pending.items()
            if future.is_resolved
        ]

        for task_id in resolved_task_ids:
            yield pending.pop(task_id)

        if not pending:
            break

        if deadline is None:
            wait_window = 30
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"{len(pending)} tasks are still pending"
                )
            wait_window = min(30, remaining)

        response = client.wait_any(
            WaitAnyRequest(
                experiment_id=experiment_id,
                task_ids=list(pending.keys()),
                wait_timeout_seconds=wait_window,
                max_results=min(32, len(pending)),
            )
        )

        # 本次 long-poll 窗口超时，继续下一轮。
        if not response.completed_tasks:
            continue

        for task_result in response.completed_tasks:
            future = pending.pop(task_result.task_id)
            future._resolve_from_remote(task_result)
            yield future
```

`as_completed()` 是 pull-based lazy resolution：只有调用方请求下一个元素时，generator 才会继续调用 `WaitAny`。每次 `yield future` 后函数暂停，外层 `for` 先处理该 Future；下一轮迭代时再从 yield 之后继续。

### RolloutAdapter 调用示例

```python
def wait_batch(self, futures):
    result_by_task_id = {}

    for future in as_completed(futures):
        # as_completed 已缓存远端结果，因此 result() 立即返回。
        result_by_task_id[future.task_id] = future.result()
        self.in_flight.discard(future)

    # as_completed 按完成顺序交付；这里恢复 submit 顺序。
    ordered_results = [
        result_by_task_id[future.task_id]
        for future in futures
    ]

    return build_verl_dataproto(ordered_results)
```

> **边界：** RolloutAdapter 只收集结果、恢复顺序并转换数据格式，不决定 `FAILED`、`ABORTED`、`TRUNCATED` 或 `CANCELLED` 样本如何参与训练；该策略由 Driver/Trainer 后续逻辑决定。
