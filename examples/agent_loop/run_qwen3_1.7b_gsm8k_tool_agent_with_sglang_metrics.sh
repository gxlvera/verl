#!/usr/bin/env bash
# Run one GSM8K tool-agent experiment and sample SGLang /metrics locally.

set -euo pipefail

LOG_ROOT=${LOG_ROOT:-/root/logs}
mkdir -p "${LOG_ROOT}"

GSM8K_TOOL_SLEEP_MS=${GSM8K_TOOL_SLEEP_MS:-600}
GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-20}
SGLANG_METRICS_INTERVAL=${SGLANG_METRICS_INTERVAL:-0.1}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_tool_agent_sglang_sleep${GSM8K_TOOL_SLEEP_MS}ms_toolcalls${GSM8K_MIN_TOOL_CALLS}_sgmetrics_$(date +%Y%m%d_%H%M%S)}
RUN_LOG_FILE=${RUN_LOG_FILE:-${LOG_ROOT}/${EXPERIMENT_NAME}.log}
SGLANG_METRICS_FILE=${SGLANG_METRICS_FILE:-${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics.jsonl}
SGLANG_METRICS_SAMPLER_LOG=${SGLANG_METRICS_SAMPLER_LOG:-${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics_sampler.log}

SAMPLER_PID=""
RUN_PID=""

cleanup() {
    if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
        kill "${SAMPLER_PID}" 2>/dev/null || true
        wait "${SAMPLER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${RUN_PID}" ]] && kill -0 "${RUN_PID}" 2>/dev/null; then
        kill "${RUN_PID}" 2>/dev/null || true
        wait "${RUN_PID}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

echo "Experiment: ${EXPERIMENT_NAME}"
echo "Run log: ${RUN_LOG_FILE}"
echo "SGLang metrics jsonl: ${SGLANG_METRICS_FILE}"
echo "SGLang metrics sampler log: ${SGLANG_METRICS_SAMPLER_LOG}"
echo "Sampling interval: ${SGLANG_METRICS_INTERVAL}s"

(
    export EXPERIMENT_NAME
    export RUN_LOG_FILE
    export LOG_ROOT
    export LOG_TO_FILE=1
	    export GSM8K_TOOL_SLEEP_MS
	    export GSM8K_MIN_TOOL_CALLS
	    export HOTPOT_TOOL_SLEEP_MS
	    export HOTPOT_TOOL_SLEEP_DIST
	    export HOTPOT_TOOL_SLEEP_SEED
	    export HOTPOT_TOOL_LATENCY_SECONDS_LIST
	    export HOTPOT_MIN_TOOL_CALLS
	    export HOTPOT_AUTO_TOOL_NAME
	    export HOTPOT_SPECULATIVE_TOOL_PREFETCH
	    export HOTPOT_SPECULATIVE_JSONL
	    export HOTPOT_MAIN_ENABLE_THINKING
	    export HOTPOT_NON_THINKING_MAX_NEW_TOKENS
	    export AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH
	    export PER_TURN_MAX_RESPONSE_LENGTH
	    export MAX_MODEL_LEN
	    export MAX_TOOL_RESPONSE_LENGTH
	    export MAX_RESPONSE_LENGTH
	    export MAX_PROMPT_LENGTH
	    export DATA_DIR
	    export TOOL_FILE
	    export TRAIN_BATCH_SIZE
	    export PPO_MINI_BATCH_SIZE
	    export PPO_MICRO_BATCH_SIZE_PER_GPU
	    export ROLLOUT_N
	    export ROLLOUT_DO_SAMPLE
	    export ROLLOUT_TEMPERATURE
	    export ROLLOUT_TOP_P
	    export ROLLOUT_TOP_K
	    export DATA_SHUFFLE
	    export DATA_SEED
	    export DATA_ENABLE_THINKING
	    export ACTOR_SHUFFLE
	    export ACTOR_DATA_LOADER_SEED
	    export CUDA_VISIBLE_DEVICES
	    export NGPUS_PER_NODE
	    export TOTAL_STEPS
	    export PROFILE_ROLLOUT_ONLY
	    export SGLANG_PROMETHEUS_ENABLE=true
    bash examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent.sh "$@"
) &
RUN_PID=$!

python3 examples/agent_loop/sample_sglang_metrics.py \
    --log-file "${RUN_LOG_FILE}" \
    --output "${SGLANG_METRICS_FILE}" \
    --interval "${SGLANG_METRICS_INTERVAL}" \
    >"${SGLANG_METRICS_SAMPLER_LOG}" 2>&1 &
SAMPLER_PID=$!

set +e
wait "${RUN_PID}"
RUN_STATUS=$?
set -e

if [[ -n "${SAMPLER_PID}" ]] && kill -0 "${SAMPLER_PID}" 2>/dev/null; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
fi

echo "Run exited with status ${RUN_STATUS}"
echo "SGLang metrics written to ${SGLANG_METRICS_FILE}"
exit "${RUN_STATUS}"
