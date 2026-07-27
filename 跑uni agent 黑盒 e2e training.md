# 跑uni agent 黑盒 e2e training

1. ~~在verl-megatron-sglang里面安装uni agent需要的依赖 ✅~~
2. 当前镜像sgl版本是否支持qwen3.5 4b？回答：支持了
3. 为什么sgl读不到正确的eos呢？

要check的点：

1. 模型是否开thinking，以及claude code发来的request会不会drop掉之前轮的thinking
2. 如果超过了cladue code max token会怎么样？直接return为truncated吗？
3. 之前为啥OOM了
4. 500 个任务中 sample 338 因 sandbox 异常未生成 trajectory，因此未计入平均值，为什么会有sandbox异常。
5. 多node trial的megatron 版本和sgl是否能对上？精度有没有问题，要不还是用vllm？

训练涨点

1. 感觉swe gym作为训练数据会不会更好？因为更集中，容易收敛

如何提速rollout

1. 换个机器
2. 换个harness
3. profile下目前的瓶颈
4. 试试提前识别claude code会丢弃的traj 然后不做这些的rollout
5. 看下任务速度跟并发量的关系
6. 看下max turn（但是jiajun说100多轮是正常的）
7. 把ctx lenght减小
8. 大模型会不会反而更快因为轮数少

maybe会出问题的点

1. 官方是跑的vllm，而我跑的sglang
2. “现有环境中 `sglang 0.5.9` 要求 `grpcio>=1.78.0`，但镜像当前是 `1.59.5`，存在潜在运行冲突。”

TODO

1. 对齐megatron和sglang版本，先用小实验测下精度，vllm开发机咋不行了
2. 开发机把codex e2e swe re bench training跑起来，超参数跟polar对比：

- [x] 训练集：当前 SWE-ReBench 1150 条；对方 SkyRL 293 条。

- [x] 训练步数：当前固定 35；对方约 37。

- [x] Codex：当前 `0.145.0`；对方 `0.125.0`。

- [x] `reasoning_effort=xhigh`：当前没有显式配置。

- [ ] TIS：对方开启；当前 rollout correction 默认关闭。

- [x] 异步：当前 `colocate_async`、warmup=1、off-policy threshold=8；没有 `async_level=2` 和 `min_complete_fraction=0.6`，默认等待全部 session 结束。

- [ ] trajectory：当前选 `longest`；对方做 `prefix_merging`。

- [x] GPU：当前 actor/rollout tp = 1 ；对方独立 4+4 卡，actor TP=2、rollout TP=1。

- [ ] 每卡 token budget：当前 48,000；对方 30,000。

- [ ] **Context length不一样，因为swe gym和swe re bench难度不同？**

- [ ] Polar 开thd（因为slime支持了qwen3.5 thd), verl用bhsd

- [ ] polar是4*16 bsz, 我是32*8 with mini batch = 16*8

- [x] SGLang 显存比例：当前 0.7；对方 0.8。

- [x] 路由：当前 least-loaded；对方 round-robin。

- [x] checkpoint：当前每 5 步；对方每 10 步。

- [x] CP：两者都是 `CP=1`，即未启用；当前脚本已显式锁定。

- [x] Codex 两个参数：危险权限参数原本已开，`unified_exec` 已补上，现在两个都开启。

- [x] Dynamic batching现在都是true了

1. 如果oom，想想办法，要不要开tp，然后多机，看下vllm那个开发机有没有weight sync问题
2. Swe gym
3. 问yuyang 问题
4. 写代码
5. 多node跑claude code，对齐超参数
6. modal
7. neo的RayPPOTrainer是否支持qwen3.5 4b

vllm开发机setup

1. yuyang那边的依赖
2. 用thd而不是bhsd
3. Gateway ip暴露
4. Psm
5. Eos
6. 脚本里改成用vllm
