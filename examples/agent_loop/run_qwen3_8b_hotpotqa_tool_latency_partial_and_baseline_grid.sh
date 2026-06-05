#!/usr/bin/env bash
# Run HotpotQA Qwen3-8B tool-agent latency grids with and without partial_async.

set -euo pipefail

LOG_ROOT=${LOG_ROOT:-/root/logs}
mkdir -p "${LOG_ROOT}"

QWEN3_8B_SNAPSHOT=${QWEN3_8B_SNAPSHOT:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
DATA_DIR=${DATA_DIR:-/root/data/hotpotqa_tool_agent}
SLEEP_MS_GRID=${SLEEP_MS_GRID:-"0 400 800 1200 1600"}
MODE_GRID=${MODE_GRID:-"baseline partial_async"}
RUN_ID=${RUN_ID:-qwen3_8b_hotpotqa_group_boundary_$(date +%Y%m%d_%H%M%S)}

if [[ ! -d "${DATA_DIR}" || ! -s "${DATA_DIR}/train.parquet" || ! -s "${DATA_DIR}/test.parquet" ]]; then
    PYTHONPATH="$(pwd)" python3 examples/data_preprocess/hotpotqa_tool_agent_loop.py \
        --local_save_dir "${DATA_DIR}"
fi

export MODEL_PATH=${MODEL_PATH:-${QWEN3_8B_SNAPSHOT}}
export DATA_DIR
export TOOL_FILE=${TOOL_FILE:-examples/agent_loop/hotpot_fixed_context_tool.py}
export HOTPOT_AUTO_TOOL_NAME=${HOTPOT_AUTO_TOOL_NAME:-retrieve_hotpot_context}
export HOTPOT_MIN_TOOL_CALLS=${HOTPOT_MIN_TOOL_CALLS:-2}
export GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-${HOTPOT_MIN_TOOL_CALLS}}
export AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH=${AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH:-500}
export PER_TURN_MAX_RESPONSE_LENGTH=${PER_TURN_MAX_RESPONSE_LENGTH:-500}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-12000}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-192}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export MOCK_BATCH_SIZE=${MOCK_BATCH_SIZE:-768}
export SGLANG_METRICS_INTERVAL=${SGLANG_METRICS_INTERVAL:-0.1}
export PARTIAL_ASYNC_NUM_STARTS=${PARTIAL_ASYNC_NUM_STARTS:-9}
export PARTIAL_ASYNC_WARMUP_STEPS=${PARTIAL_ASYNC_WARMUP_STEPS:-0}

export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export DATA_SEED=${DATA_SEED:-42}
export ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-False}
export ACTOR_DATA_LOADER_SEED=${ACTOR_DATA_LOADER_SEED:-42}
export ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-False}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}

echo "Running HotpotQA Qwen3-8B grid: modes=${MODE_GRID}, sleeps=${SLEEP_MS_GRID}, run_id=${RUN_ID}"
echo "DATA_DIR=${DATA_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "Config: train_batch=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}, total_steps=${TOTAL_STEPS}, per_turn=${PER_TURN_MAX_RESPONSE_LENGTH}, max_response=${MAX_RESPONSE_LENGTH}, max_model_len=${MAX_MODEL_LEN}, mock_batch=${MOCK_BATCH_SIZE}, partial_async_num_starts=${PARTIAL_ASYNC_NUM_STARTS}, partial_async_warmup_steps=${PARTIAL_ASYNC_WARMUP_STEPS}"

for mode in ${MODE_GRID}; do
    for sleep_ms in ${SLEEP_MS_GRID}; do
        export HOTPOT_TOOL_SLEEP_MS="${sleep_ms}"
        export GSM8K_TOOL_SLEEP_MS="${sleep_ms}"
        export HOTPOT_TOOL_SLEEP_DIST=""
        export GSM8K_TOOL_SLEEP_DIST=""
        export HOTPOT_TOOL_SLEEP_SEED=""
        export GSM8K_TOOL_SLEEP_SEED=""

        if [[ "${mode}" == "partial_async" ]]; then
            export GSM8K_AGENT_LOOP_MODE=partial_async
            export PARTIAL_ASYNC_NUM_STARTS
            export PARTIAL_ASYNC_WARMUP_STEPS
            unset GSM8K_PARTIAL_ASYNC
        else
            export GSM8K_AGENT_LOOP_MODE=
            unset GSM8K_PARTIAL_ASYNC
        fi

        export EXPERIMENT_NAME="${mode}_hotpotqa_qwen3_8b_sleep${sleep_ms}_toolcalls${HOTPOT_MIN_TOOL_CALLS}_bs${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_step${TOTAL_STEPS}_${RUN_ID}"
        export RUN_LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}.log"
        export SGLANG_METRICS_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics.jsonl"
        export SGLANG_METRICS_SAMPLER_LOG="${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics_sampler.log"
        export PARTIAL_ASYNC_JSONL="${LOG_ROOT}/${EXPERIMENT_NAME}_partial_async.jsonl"

        echo "===== Running ${EXPERIMENT_NAME} ====="
        set +e
        bash examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent_with_sglang_metrics.sh
        status=$?
        set -e

        if [[ -s "${SGLANG_METRICS_FILE}" ]]; then
            python3 examples/agent_loop/plot_sglang_running_requests.py \
                --jsonl "${SGLANG_METRICS_FILE}" \
                --title "HotpotQA ${mode} running requests, sleep=${sleep_ms}ms" \
                >"${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log" 2>&1 || true
        fi

        if [[ "${status}" != "0" ]]; then
            echo "Run ${EXPERIMENT_NAME} exited with status ${status}; continuing."
        fi

        if command -v ray >/dev/null 2>&1; then
            ray stop --force || true
            sleep 5
        fi
    done
done

echo "Finished HotpotQA Qwen3-8B grid: ${RUN_ID}"
