# GSM8K Tool-Latency Rollout Experiments

This note summarizes the VERL + SGLang tool-call latency experiments run during
the June 2026 debugging session. The common workload is GSM8K multi-turn tool
agent rollout with GRPO (`algorithm.adv_estimator=grpo`) through the PPO trainer
entrypoint (`python3 -m verl.trainer.main_ppo`).

## Shared Setup

Base runner:

```bash
examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent.sh
```

The runner was converted to SGLang async rollout and exposes these important
controls:

```text
GSM8K_TOOL_SLEEP_MS       fixed per-tool artificial sleep in ms
GSM8K_TOOL_SLEEP_DIST     weighted sleep distribution, e.g. 0:40,300:30,600:30
GSM8K_TOOL_SLEEP_SEED     per-worker RNG seed for sleep distribution
GSM8K_MIN_TOOL_CALLS      minimum number of tool calls per trajectory
MOCK_BATCH_SIZE           rollout completion bucket size for timing logs
SGLANG_PROMETHEUS_ENABLE  enables /metrics for SGLang
SGLANG_PROMETHEUS_PORT    Prometheus metrics port
DATA_SHUFFLE              prompt order shuffle; defaults to False
DATA_SEED                 fixed data seed; defaults to 42
ROLLOUT_DO_SAMPLE         rollout sampling; defaults to False
ROLLOUT_TEMPERATURE       rollout temperature; defaults to 0
ROLLOUT_TOP_P             rollout top-p; defaults to 1.0
ROLLOUT_TOP_K             rollout top-k; defaults to -1
```

The default decoding mode is greedy (`do_sample=False`, `temperature=0`), which
removes normal token-sampling randomness. It does not guarantee bitwise identical
training runs because async scheduling, GPU kernels, and model updates can still
diverge.

## Instrumentation Added

VERL-side agent-loop metrics:

```text
timing_s/agent_loop/state/waiting_tool/{mean,max,zero_fraction}
timing_s/agent_loop/state/ready_to_llm/{mean,max,zero_fraction}
timing_s/agent_loop/state/llm_inflight/{mean,max,zero_fraction}
timing_s/agent_loop/state/ready_or_inflight/{mean,max,zero_fraction}
timing_s/agent_loop/state/tool_waiting_with_no_llm/fraction
timing_s/agent_loop/state/total_wall_time
timing_s/agent_loop/mock_batch_time/{first,last,mean_completion,32,...}
timing_s/agent_loop/slowest/{e2e,generate_sequences,tool_calls,tool_call_count,response_length}
agent_loop/tool_call_count_dist/<count>
```

These answer the question: while some samples wait for tools, does VERL still
have ready or in-flight LLM work to feed SGLang?

SGLang-side sampler and plots:

```bash
examples/agent_loop/sample_sglang_metrics.py
examples/agent_loop/plot_sglang_running_requests.py
examples/agent_loop/plot_sglang_step_zoom.py
```

The sampler scrapes SGLang `/metrics` every `SGLANG_METRICS_INTERVAL` seconds
and records JSONL under `/root/logs`. The plots aggregate across SGLang servers:

```text
sglang:num_running_reqs
sglang:num_queue_reqs
sglang:gen_throughput
sglang:decode_sum_seq_lens
```

The zoom plot merges nearby peak fragments before selecting main rollout peaks,
so short dips inside one step do not get counted as separate steps.

## Experiment Categories

### 1. Initial Qwen3-1.7B Latency Grid

Scripts:

```bash
examples/agent_loop/run_gsm8k_tool_latency_grid.sh
examples/agent_loop/run_gsm8k_tool_latency_2calls_200_700.sh
```

Purpose:

```text
Map fixed sleep latency against minimum tool-call count.
Compare first mock-batch time, last mock-batch time, gen time, and throughput.
```

Representative settings:

```text
model: Qwen3-1.7B
sleep: 200,300,400,500,600,700ms
min tool calls: 2
rollout_n: 8
```

Early observation:

```text
Increasing tool sleep pushed first mock-batch collection later. Last mock-batch
time was often similar, but gen still changed because decode workload and tool
overlap changed across trajectories.
```

### 2. Random Tool-Latency Distribution Grid

Scripts:

```bash
examples/agent_loop/run_gsm8k_tool_latency_dist.sh
examples/agent_loop/run_gsm8k_tool_latency_dist_grid.sh
```

Requested distributions:

```text
300:30,600:70
350:30,600:70
400:30,600:70
0:40,300:30,600:30
0:60,300:30,600:10
0:80,600:20
```

Purpose:

```text
Simulate mixed tool latencies with a fixed random seed instead of a single
uniform sleep value.
```

### 3. Qwen3-1.7B SGLang Metrics Smoke Tests

Scripts:

```bash
examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent_with_sglang_metrics.sh
examples/agent_loop/run_sglang_metrics_sleep0_300_grid.sh
```

Representative local logs:

```text
/root/logs/sglang_metrics_sleep600_bs16_rollout8_step5_retry_20260603_222129_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep0_bs16_rollout8_step5_20260603_235019_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep300_bs16_rollout8_step5_20260603_235019_sglang_metrics.jsonl
```

Purpose:

```text
Validate SGLang /metrics sampling at 0.1s resolution and confirm that
running/queue request counts are meaningful.
```

Important note:

```text
SGLang queue requests measure requests already submitted to SGLang but waiting
inside its scheduler. They do not measure samples stuck in VERL waiting for
tools. For that, use VERL-side ready/waiting/inflight state metrics.
```

### 4. Qwen3-1.7B Toolcalls5, Sleep 0/300/600, Step3

Script:

```bash
examples/agent_loop/run_sglang_metrics_sleep0_300_600_toolcalls5_step3.sh
```

Representative bs16 logs:

```text
/root/logs/sglang_metrics_sleep0_toolcalls5_bs16_rollout8_step3_toolcalls5_20260604_064703_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep300_toolcalls5_bs16_rollout8_step3_toolcalls5_20260604_064703_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep600_toolcalls5_bs16_rollout8_step3_toolcalls5_20260604_064703_sglang_metrics.jsonl
```

Representative bs64 logs:

```text
/root/logs/sglang_metrics_sleep0_toolcalls5_bs64_rollout8_step3_toolcalls5_bs64_20260604_203229_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep300_toolcalls5_bs64_rollout8_step3_toolcalls5_bs64_20260604_203229_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep600_toolcalls5_bs64_rollout8_step3_toolcalls5_bs64_20260604_203229_sglang_metrics.jsonl
```

Key plot outputs:

```text
/root/logs/sglang_metrics_sleep0_toolcalls5_bs64_rollout8_step3_toolcalls5_bs64_20260604_203229_two_main_peaks_zoom_0p1s_fixed_ylim.png
/root/logs/sglang_metrics_sleep300_toolcalls5_bs64_rollout8_step3_toolcalls5_bs64_20260604_203229_two_main_peaks_zoom_0p1s_fixed_ylim.png
/root/logs/sglang_metrics_sleep600_toolcalls5_bs64_rollout8_step3_toolcalls5_bs64_20260604_203229_two_main_peaks_zoom_0p1s_fixed_ylim.png
```

Main findings:

```text
The summed SGLang running request peak reached 512/513 for bs64 * rollout_n 8.
Queue requests were essentially zero in these runs. When sleep grew, VERL-side
waiting_tool increased and ready_or_inflight zero time increased, meaning the
rollout engine was often starved by upstream tool waits rather than blocked by
SGLang queue pressure.
```

### 5. Deterministic Qwen3-1.7B Toolcalls3, Step20

Script:

```bash
examples/agent_loop/run_sglang_metrics_sleep0_300_600_toolcalls3_step20_deterministic.sh
```

Purpose:

```text
Run sleep 0/300/600ms with min tool calls 3, 20 steps, fixed prompt order, fixed
data seed, actor shuffle disabled, and greedy rollout decoding.
```

Representative logs:

```text
/root/logs/sglang_metrics_sleep0_toolcalls3_bs64_rollout8_step20_toolcalls3_step20_deterministic_20260604_215000_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep300_toolcalls3_bs64_rollout8_step20_toolcalls3_step20_deterministic_20260604_215000_sglang_metrics.jsonl
/root/logs/sglang_metrics_sleep600_toolcalls3_bs64_rollout8_step20_toolcalls3_step20_deterministic_20260604_215000_sglang_metrics.jsonl
```

Note:

```text
An earlier accidental run used toolcalls2 and bs16:
/root/logs/sglang_metrics_sleep0_toolcalls2_bs16_rollout8_step20_toolcalls3_step20_deterministic_20260604_213315_sglang_metrics.jsonl
```

### 6. Qwen3-8B Toolcalls5, Sleep0, Step3

Command pattern:

```bash
MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
SLEEP_MS_GRID="0" \
GSM8K_MIN_TOOL_CALLS=5 \
TRAIN_BATCH_SIZE=64 \
ROLLOUT_N=8 \
TOTAL_STEPS=3 \
CUDA_VISIBLE_DEVICES=0,1 \
NGPUS_PER_NODE=2 \
bash examples/agent_loop/run_sglang_metrics_sleep0_300_600_toolcalls5_step3.sh
```

Representative log:

```text
/root/logs/sglang_metrics_sleep0_toolcalls5_bs64_rollout8_step3_qwen3_8b_toolcalls5_sleep0_step3_20260605_003438_sglang_metrics.jsonl
```

Key outputs:

```text
/root/logs/sglang_metrics_sleep0_toolcalls5_bs64_rollout8_step3_qwen3_8b_toolcalls5_sleep0_step3_20260605_003438_active_replot.png
/root/logs/sglang_metrics_sleep0_toolcalls5_bs64_rollout8_step3_qwen3_8b_toolcalls5_sleep0_step3_20260605_003438_two_main_peaks_zoom_0p1s_fixed_ylim.png
```

Finding:

```text
running_max=512. Queue was nonzero only once: queue=16 at monotonic_s=206.0s.
That was the initial request surge, not sustained scheduler pressure.
```

### 7. Qwen3-8B Toolcalls2, Sleep 400/800/1200/1600, Step10

Script:

```bash
examples/agent_loop/run_qwen3_8b_sleep400_800_1200_1600_toolcalls2_step10.sh
```

Representative logs:

```text
/root/logs/sglang_metrics_qwen3_8b_sleep400_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_sglang_metrics.jsonl
/root/logs/sglang_metrics_qwen3_8b_sleep800_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_sglang_metrics.jsonl
/root/logs/sglang_metrics_qwen3_8b_sleep1200_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_sglang_metrics.jsonl
/root/logs/sglang_metrics_qwen3_8b_sleep1600_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_sglang_metrics.jsonl
```

Selected plot outputs:

```text
/root/logs/sglang_metrics_qwen3_8b_sleep400_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_active_replot.png
/root/logs/sglang_metrics_qwen3_8b_sleep800_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_active_replot.png
/root/logs/sglang_metrics_qwen3_8b_sleep1200_toolcalls2_bs64_rollout8_step10_qwen3_8b_toolcalls2_step10_20260605_004918_active_replot.png
```

Observed queue pressure:

```text
sleep400:  queue_max=0
sleep800:  queue_max=60, nonzero for 2 samples at 213.8s and 213.9s
sleep1200: queue_max=76, nonzero for 1 sample
```

Interpretation:

```text
Queue pressure remained negligible. The later surprise that sleep1200 sometimes
had smaller gen than sleep400/800 was explained by shorter generated responses
and fewer 2048-token decode tails in sleep1200, not by better overlap.
```

## How To Read The Main Metrics

```text
ready_or_inflight/zero_fraction
  Fraction of sampled agent-loop time where ready_to_llm + llm_inflight == 0.
  Higher means the rollout LLM often had no ready work.

tool_waiting_with_no_llm/fraction
  Fraction of time where at least one sample waited for a tool and no sample was
  ready or in-flight at the LLM. This is direct evidence of unhidden tool wait.

waiting_tool/mean
  Average number of worker-local samples waiting for tool results. This is a raw
  count, not normalized by batch size.

sglang:num_queue_reqs
  Requests already submitted to SGLang but not yet admitted by its scheduler.
  Near-zero queue means SGLang itself is not the bottleneck.

sglang:num_running_reqs
  Requests actively running in SGLang. For bs64 and rollout_n=8, the maximum
  expected submitted sample count per step is 512.
```

## Working Hypothesis

The experiments support the following picture:

```text
Long tool sleep is not only a final-tail problem. Increasing sleep can delay the
first mock batch, which means the system can be starved in the middle of rollout
when many samples are waiting for tools and too few samples are ready to decode.

SGLang queue is almost always empty in the observed runs, so the primary issue is
upstream supply of ready LLM work, not SGLang scheduler capacity.

Comparing gen across runs must be normalized by response length and long-tail
decode behavior. A run with longer tool sleep can still show smaller gen if it
has fewer samples decoding to max_response_length.
```
