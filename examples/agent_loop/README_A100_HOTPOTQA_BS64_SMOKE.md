# A100 HotpotQA Qwen3-8B BS64 Smoke Test

This note is for running a small HotpotQA + Qwen3-8B SGLang rollout stress test
on an A100 machine. The goal is to check whether the workload
`train_batch_size=64, rollout_n=8` can fit and run normally.

## Goal

Run baseline rollout only, no partial async:

```text
model: Qwen3-8B
dataset: HotpotQA tool-agent parquet
train batch size: 64
rollout_n: 8
expanded rollout samples per step: 64 * 8 = 512
tool call latency: 0ms, 400ms, 800ms
minimum tool calls per trajectory: 2
steps per experiment: 3
GPUs: 8
tool response: fixed synthetic 500-token evidence text
```

## Script

Use the existing grid script:

```bash
examples/agent_loop/run_qwen3_8b_hotpotqa_tool_latency_partial_and_baseline_grid.sh
```

It will also generate SGLang running/queue request plots after each run via:

```bash
examples/agent_loop/plot_sglang_running_requests.py
```

## Command

Change `MODEL_PATH` to the actual Qwen3-8B path on the A100 machine.

```bash
cd /root/verl

LOG_ROOT=/home/tiger \
MODEL_PATH=<A100机器上的Qwen3-8B模型路径> \
DATA_DIR=/root/data/hotpotqa_tool_agent \
SLEEP_MS_GRID="0 400 800" \
MODE_GRID="baseline" \
TRAIN_BATCH_SIZE=64 \
ROLLOUT_N=8 \
TOTAL_STEPS=3 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NGPUS_PER_NODE=8 \
HOTPOT_MIN_TOOL_CALLS=2 \
GSM8K_MIN_TOOL_CALLS=2 \
PER_TURN_MAX_RESPONSE_LENGTH=500 \
AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH=500 \
MAX_RESPONSE_LENGTH=4096 \
MAX_MODEL_LEN=4096 \
MAX_TOOL_RESPONSE_LENGTH=12000 \
MAX_PROMPT_LENGTH=4096 \
MOCK_BATCH_SIZE=128 \
SGLANG_METRICS_INTERVAL=0.1 \
DATA_SHUFFLE=False \
DATA_SEED=42 \
ACTOR_SHUFFLE=False \
ACTOR_DATA_LOADER_SEED=42 \
ROLLOUT_DO_SAMPLE=False \
ROLLOUT_TEMPERATURE=0 \
ROLLOUT_TOP_P=1.0 \
ROLLOUT_TOP_K=-1 \
RUN_ID="qwen3_8b_hotpotqa_bs64_toolcalls2_a100_smoke_$(date +%Y%m%d_%H%M%S)" \
bash examples/agent_loop/run_qwen3_8b_hotpotqa_tool_latency_partial_and_baseline_grid.sh
```

## tmux

Interactive tmux:

```bash
tmux new -s a100-bs64-hotpot
```

Then paste the command above.

## Data

If the HotpotQA parquet data does not exist, the script should generate it
automatically. To generate it manually:

```bash
cd /root/verl

PYTHONPATH="$(pwd)" python3 examples/data_preprocess/hotpotqa_tool_agent_loop.py \
  --local_save_dir /root/data/hotpotqa_tool_agent
```

Expected files:

```text
/root/data/hotpotqa_tool_agent/train.parquet
/root/data/hotpotqa_tool_agent/test.parquet
```

## Outputs

Logs and plots will be under:

```text
/home/tiger
```

Expected files include:

```text
/home/tiger/baseline_hotpotqa_qwen3_8b_sleep0_toolcalls2_bs64_rollout8_step3_<RUN_ID>.log
/home/tiger/baseline_hotpotqa_qwen3_8b_sleep0_toolcalls2_bs64_rollout8_step3_<RUN_ID>_sglang_metrics.jsonl
/home/tiger/baseline_hotpotqa_qwen3_8b_sleep0_toolcalls2_bs64_rollout8_step3_<RUN_ID>_sglang_metrics_running_reqs_active.png

/home/tiger/baseline_hotpotqa_qwen3_8b_sleep400_toolcalls2_bs64_rollout8_step3_<RUN_ID>_sglang_metrics_running_reqs_active.png
/home/tiger/baseline_hotpotqa_qwen3_8b_sleep800_toolcalls2_bs64_rollout8_step3_<RUN_ID>_sglang_metrics_running_reqs_active.png
```

## What To Check

Watch for:

```text
1. OOM or SGLang startup failure.
2. Whether SGLang running requests reaches the expected workload scale.
3. Whether queue requests are only short admission bursts or stay nonzero.
4. W&B timing_s/gen, throughput, and GPU utilization.
5. Local running request plot PNGs under /home/tiger.
```

Important distinction:

```text
SGLang queue requests only mean requests already reached SGLang but are waiting
inside its scheduler. Samples waiting on tool latency inside VERL are not visible
to SGLang queue metrics.
```
