# HotpotQA Tool-Latency And Partial-Async Handoff

This README describes the current local changes for the VERL + SGLang tool
latency experiments. It is meant for moving this repo to another server or for a
new coding agent to continue the work without replaying the whole debugging
conversation.

## Current Branch State

The branch is `toolSpec_profiling`. Earlier commits already added the GSM8K
tool-latency instrumentation and SGLang metrics plotting helpers. The latest
changes extend that setup to Qwen3-8B HotpotQA experiments, partial async
rollout, and standalone SGLang pressure tests.

Generated files such as `/root/logs`, `tensorboard_log/`, and `core` dumps are
not part of the experiment code and should not be committed.

## Code Changes

### 1. Shared Runner Extensions

Files:

```text
examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent.sh
examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent_with_sglang_metrics.sh
```

Despite the filename, these scripts are now used as the common runner for both
GSM8K and HotpotQA, including Qwen3-8B. Important environment variables:

```text
MODEL_PATH                         HuggingFace model path.
DATA_DIR                           Directory containing train.parquet/test.parquet.
TOOL_FILE                          Python tool file loaded by VERL.
TRAIN_BATCH_SIZE                   Prompt batch size before rollout_n expansion.
ROLLOUT_N                          Number of rollout samples per prompt.
TOTAL_STEPS                        Training steps.
CUDA_VISIBLE_DEVICES               Visible GPUs.
NGPUS_PER_NODE                     Number of GPUs used by VERL.
MAX_PROMPT_LENGTH                  Prompt truncation limit.
MAX_RESPONSE_LENGTH                Full trajectory response budget.
PER_TURN_MAX_RESPONSE_LENGTH       Optional max_new_tokens cap per assistant turn.
MAX_MODEL_LEN                      SGLang context length override.
MAX_TOOL_RESPONSE_LENGTH           Tool response truncation budget.
SGLANG_METRICS_INTERVAL            /metrics scrape interval, usually 0.1.
```

HotpotQA-specific variables:

```text
HOTPOT_TOOL_SLEEP_MS               Fixed artificial sleep per tool call.
HOTPOT_TOOL_SLEEP_DIST             Weighted sleep distribution, e.g. 0:40,300:30,600:30.
HOTPOT_TOOL_SLEEP_SEED             Seed for weighted sleep sampling.
HOTPOT_MIN_TOOL_CALLS              Minimum forced tool calls.
HOTPOT_AUTO_TOOL_NAME              Usually retrieve_hotpot_context.
```

Determinism variables used in the recent experiments:

```text
DATA_SHUFFLE=False
DATA_SEED=42
ACTOR_SHUFFLE=False
ACTOR_DATA_LOADER_SEED=42
ROLLOUT_DO_SAMPLE=False
ROLLOUT_TEMPERATURE=0
ROLLOUT_TOP_P=1.0
ROLLOUT_TOP_K=-1
```

This makes rollout decoding greedy, but it does not guarantee bitwise-identical
training because async scheduling and GPU kernels can still vary.

### 2. HotpotQA Data And Tool

Files:

```text
examples/data_preprocess/hotpotqa_tool_agent_loop.py
examples/agent_loop/hotpot_fixed_context_tool.py
```

Prepare HotpotQA parquet data:

```bash
cd /root/verl
PYTHONPATH="$(pwd)" python3 examples/data_preprocess/hotpotqa_tool_agent_loop.py \
  --local_save_dir /root/data/hotpotqa_tool_agent
```

The preprocess script writes:

```text
/root/data/hotpotqa_tool_agent/train.parquet
/root/data/hotpotqa_tool_agent/test.parquet
```

The HotpotQA tool is `retrieve_hotpot_context`. It returns a fixed synthetic
500-token evidence string and optionally sleeps. Sleep can be fixed or sampled
from a weighted distribution.

### 3. VERL Partial Async Mode

Files:

```text
verl/experimental/agent_loop/agent_loop.py
verl/experimental/agent_loop/tool_agent_loop.py
verl/trainer/ppo/metric_utils.py
```

Enable partial async with:

```text
GSM8K_AGENT_LOOP_MODE=partial_async
```

or:

```text
GSM8K_PARTIAL_ASYNC=1
```

The mode is implemented in `AgentLoopManager._generate_sequences_partial_async`.
It is currently intended for `tool_agent` experiments.

Behavior:

```text
1. Start with the normal expanded rollout batch.
2. Group samples by prompt group, where group size is rollout_n.
3. When half of the prompt groups finish, record a boundary.
4. Immediately launch another half-batch of extra samples.
5. Repeat for PARTIAL_ASYNC_NUM_STARTS starts.
6. Only start_t=0 samples are returned for training.
7. Extra samples are used only to keep SGLang busy and to log boundary timing.
```

Important partial async variables:

```text
PARTIAL_ASYNC_NUM_STARTS           Number of start waves. Default 5.
PARTIAL_ASYNC_THRESHOLD            Finished prompt groups per boundary. Default half of initial groups.
PARTIAL_ASYNC_WARMUP_STEPS         Steps to skip partial_async. Default 1.
PARTIAL_ASYNC_JSONL                Output JSONL path. Defaults under LOG_ROOT.
```

Metrics logged to W&B are intentionally not prefixed with `timing_s/`:

```text
partial_async/enabled
partial_async/initial_samples
partial_async/initial_groups
partial_async/rollout_n
partial_async/threshold
partial_async/threshold_groups
partial_async/num_starts
partial_async/total_samples
partial_async/extra_samples
partial_async/wall_time
partial_async/boundary/<n>/relative_s
partial_async/boundary/<n>/epoch_s
partial_async/boundary/<n>/count
partial_async/boundary/<n>/group_count
```

Each partial async step also appends detailed per-sample records to
`PARTIAL_ASYNC_JSONL`, including:

```text
step
sample_id
source_index
group_id
rollout_index
start_t
end_t
is_train
submit_relative_s
complete_relative_s
boundary_relative_s
response_length
num_turns
tool_call_count
```

### 4. HotpotQA Baseline And Partial-Async Grid

File:

```text
examples/agent_loop/run_qwen3_8b_hotpotqa_tool_latency_partial_and_baseline_grid.sh
```

This is the main handoff script for the recent Qwen3-8B HotpotQA comparison.
It runs baseline and partial async experiments over a sleep grid and samples
SGLang metrics every 0.1s.

Example run:

```bash
cd /root/verl

tmux new -s hotpotqa-grid

LOG_ROOT=/root/logs \
MODEL_PATH=/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
DATA_DIR=/root/data/hotpotqa_tool_agent \
SLEEP_MS_GRID="0 400 800 1200 1600" \
MODE_GRID="baseline partial_async" \
TRAIN_BATCH_SIZE=128 \
ROLLOUT_N=8 \
TOTAL_STEPS=10 \
CUDA_VISIBLE_DEVICES=0,1 \
NGPUS_PER_NODE=2 \
HOTPOT_MIN_TOOL_CALLS=2 \
PER_TURN_MAX_RESPONSE_LENGTH=500 \
MAX_RESPONSE_LENGTH=4096 \
MAX_MODEL_LEN=4096 \
MAX_TOOL_RESPONSE_LENGTH=12000 \
PARTIAL_ASYNC_NUM_STARTS=9 \
PARTIAL_ASYNC_WARMUP_STEPS=0 \
bash examples/agent_loop/run_qwen3_8b_hotpotqa_tool_latency_partial_and_baseline_grid.sh
```

The script creates logs such as:

```text
/root/logs/<EXPERIMENT_NAME>.log
/root/logs/<EXPERIMENT_NAME>_sglang_metrics.jsonl
/root/logs/<EXPERIMENT_NAME>_sglang_metrics_running_reqs_active.csv
/root/logs/<EXPERIMENT_NAME>_sglang_metrics_running_reqs_active.png
/root/logs/<EXPERIMENT_NAME>_partial_async.jsonl
```

If moving to another server, check the Qwen3-8B snapshot path first. The default
in the script points to the local path used on the original machine.

### 5. Older GSM8K Partial-Async Grid

File:

```text
examples/agent_loop/run_qwen3_8b_partial_async_sleep400_800_1200_1600_toolcalls3_step4.sh
```

This runs Qwen3-8B on the original GSM8K tool-agent workload with:

```text
sleep: 400,800,1200,1600ms
min tool calls: 3
total steps: 4
partial_async enabled
```

Use it mainly as a reference or smoke test. The newer HotpotQA grid above is the
better entry point for the current analysis.

### 6. Standalone SGLang Pressure Test

File:

```text
examples/agent_loop/sglang_hotpot_pressure_test.py
```

This script starts an SGLang server, samples `/metrics`, sends concurrent
HotpotQA requests, and plots scheduler behavior. It does not run VERL training.

Single-turn mode:

```bash
cd /root/verl

python3 examples/agent_loop/sglang_hotpot_pressure_test.py \
  --mode single \
  --gpu 2 \
  --num-prompts 128 \
  --extra-tokens 2000 \
  --max-new-tokens 1000 \
  --metrics-interval 0.1 \
  --run-name hotpot_qwen3_8b_gpu2_128req_single
```

Multi-turn search simulation:

```bash
cd /root/verl

python3 examples/agent_loop/sglang_hotpot_pressure_test.py \
  --mode multi_turn_search \
  --gpu 2 \
  --num-prompts 512 \
  --initial-prompt-tokens 300 \
  --tool-response-tokens 500 \
  --assistant-turns 3 \
  --max-new-tokens 1000 \
  --context-length 8192 \
  --max-running-requests 1000 \
  --metrics-interval 0.1 \
  --run-name hotpot_qwen3_8b_gpu2_512req_multiturn_maxrun1000
```

Multi-turn behavior:

```text
1. Initial prompt is about 300 tokens.
2. Assistant generates turn 1.
3. Script appends a synthetic 500-token tool/search response.
4. Assistant generates turn 2.
5. Script appends another unique synthetic 500-token tool/search response.
6. Assistant generates turn 3.
```

The injected tool response is unique per request and per tool turn. The result
JSONL logs `injected_tool_response_seed`.

Pressure-test outputs:

```text
/root/logs/<RUN_NAME>_server.log
/root/logs/<RUN_NAME>_prompts.jsonl
/root/logs/<RUN_NAME>_results.jsonl
/root/logs/<RUN_NAME>_sglang_metrics.jsonl
/root/logs/<RUN_NAME>_schedule.csv
/root/logs/<RUN_NAME>_schedule.png
/root/logs/<RUN_NAME>_summary.json
```

Useful settings:

```text
--max-running-requests 1000       Avoid the default 128 cap when stress-testing admission.
--disable-cuda-graph-server       Saves memory but changes the workload; recent tests prefer CUDA graph on.
--mem-fraction-static 0.82        Adjust if SGLang fails to start from memory pressure.
--start-server/--no-start-server  Use an existing SGLang server if needed.
--stop-server/--no-stop-server    Keep server alive after the test if needed.
```

## Plotting And Analysis Helpers

Already tracked helper scripts:

```text
examples/agent_loop/sample_sglang_metrics.py
examples/agent_loop/plot_sglang_running_requests.py
examples/agent_loop/plot_sglang_step_zoom.py
```

Typical active-window plot:

```bash
python3 examples/agent_loop/plot_sglang_running_requests.py \
  --jsonl /root/logs/<EXPERIMENT_NAME>_sglang_metrics.jsonl \
  --title "SGLang running requests"
```

Typical zoom plot:

```bash
python3 examples/agent_loop/plot_sglang_step_zoom.py \
  --jsonl /root/logs/<EXPERIMENT_NAME>_sglang_metrics.jsonl \
  --output-png /root/logs/<EXPERIMENT_NAME>_step2_3_zoom_0p1s.png \
  --steps 1,2
```

For later steps, it is often better to parse active segments from the generated
CSV and select the segments corresponding to the desired rollout steps. The CSV
columns are:

```text
monotonic_s
running_reqs_sum
queue_reqs_sum
gen_throughput_sum
decode_sum_seq_lens_sum
num_servers_sampled
```

## Metrics Interpretation

VERL-side metrics answer whether rollout has ready LLM work:

```text
timing_s/agent_loop/state/waiting_tool/mean
  Average number of local samples waiting for tool results.

timing_s/agent_loop/state/ready_or_inflight/zero_fraction
  Fraction of agent-loop sampled time where ready_to_llm + llm_inflight == 0.

timing_s/agent_loop/state/tool_waiting_with_no_llm/fraction
  Fraction of time where some sample is waiting for tools and no sample is ready
  or in flight at the LLM. This is direct evidence that tool wait is not hidden.
```

SGLang-side metrics answer whether SGLang itself is saturated or queuing:

```text
sglang:num_running_reqs
  Requests actively admitted/running inside SGLang.

sglang:num_queue_reqs
  Requests already sent to SGLang but waiting in the SGLang scheduler queue.

sglang:gen_throughput
  SGLang generation throughput.

sglang:decode_sum_seq_lens
  Aggregate decode sequence length pressure.
```

Important distinction:

```text
SGLang queue == 0 does not mean VERL has ready samples. It only means requests
that already reached SGLang are not queued. Samples waiting for tool results are
in VERL and invisible to SGLang.
```

## Practical Migration Checklist

1. Clone or copy the repo and checkout `toolSpec_profiling`.
2. Make sure Qwen3-8B exists locally or set `MODEL_PATH` to the new snapshot.
3. Install dependencies needed by VERL, SGLang, `datasets`, `aiohttp`,
   `matplotlib`, and `transformers`.
4. Generate HotpotQA parquet data with
   `examples/data_preprocess/hotpotqa_tool_agent_loop.py`.
5. Start from the HotpotQA grid script with a tiny smoke test:

```bash
SLEEP_MS_GRID="0" MODE_GRID="baseline" TOTAL_STEPS=1 TRAIN_BATCH_SIZE=8 ROLLOUT_N=2 \
bash examples/agent_loop/run_qwen3_8b_hotpotqa_tool_latency_partial_and_baseline_grid.sh
```

6. Confirm these files appear under `/root/logs`:

```text
*_sglang_metrics.jsonl
*_sglang_metrics_running_reqs_active.png
*.log
```

7. Run the full desired grid only after the smoke test succeeds.

## Known Caveats

The partial async implementation launches extra samples only for measurement.
Only the original `start_t=0` samples are returned to training. This is by
design for the current experiment.

`PARTIAL_ASYNC_THRESHOLD` is in prompt groups, not raw rollout samples. With
`TRAIN_BATCH_SIZE=128` and `ROLLOUT_N=8`, the expanded initial batch is 1024
samples and the default threshold is 64 prompt groups, i.e. 512 samples.

For `TRAIN_BATCH_SIZE=192` and `ROLLOUT_N=8`, the expanded initial batch is
1536 samples and the default threshold is 96 prompt groups, i.e. 768 samples.

The runner still has `qwen3_1.7b_gsm8k` in its filename because it was the
original entrypoint. Use `MODEL_PATH`, `DATA_DIR`, and `TOOL_FILE` to switch the
actual workload.

Do not compare `timing_s/gen` without also checking response length and decode
tails. Longer tool sleep can occasionally show smaller gen if that run generated
shorter responses or fewer max-length tails.
