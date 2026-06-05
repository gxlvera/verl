#!/usr/bin/env bash
# Run the GSM8K partial_async experiment with Qwen3-8B.

set -euo pipefail

LOG_ROOT=${LOG_ROOT:-/root/logs}
mkdir -p "${LOG_ROOT}"

QWEN3_8B_SNAPSHOT=${QWEN3_8B_SNAPSHOT:-/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}
SLEEP_MS_GRID=${SLEEP_MS_GRID:-"400 800 1200 1600"}
RUN_ID=${RUN_ID:-qwen3_8b_partial_async_toolcalls3_step4_$(date +%Y%m%d_%H%M%S)}

export MODEL_PATH=${MODEL_PATH:-${QWEN3_8B_SNAPSHOT}}
export GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-3}
export GSM8K_AGENT_LOOP_MODE=${GSM8K_AGENT_LOOP_MODE:-partial_async}
export PARTIAL_ASYNC_NUM_STARTS=${PARTIAL_ASYNC_NUM_STARTS:-5}
export SGLANG_METRICS_INTERVAL=${SGLANG_METRICS_INTERVAL:-0.1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

export DATA_SHUFFLE=${DATA_SHUFFLE:-False}
export DATA_SEED=${DATA_SEED:-42}
export ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-False}
export ACTOR_DATA_LOADER_SEED=${ACTOR_DATA_LOADER_SEED:-42}
export ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-False}
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

echo "Running Qwen3-8B partial_async grid: sleeps=${SLEEP_MS_GRID}, min_tool_calls=${GSM8K_MIN_TOOL_CALLS}, total_steps=${TOTAL_STEPS}, run_id=${RUN_ID}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "Partial async config: threshold=${PARTIAL_ASYNC_THRESHOLD:-auto_half_initial_batch}, num_starts=${PARTIAL_ASYNC_NUM_STARTS}; step1 is warmup"
echo "Common config: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, ROLLOUT_N=${ROLLOUT_N}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NGPUS_PER_NODE=${NGPUS_PER_NODE}"

for sleep_ms in ${SLEEP_MS_GRID}; do
    export GSM8K_TOOL_SLEEP_MS="${sleep_ms}"
    export EXPERIMENT_NAME="partial_async_qwen3_8b_sleep${sleep_ms}_toolcalls${GSM8K_MIN_TOOL_CALLS}_bs${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_step${TOTAL_STEPS}_${RUN_ID}"
    export RUN_LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}.log"
    export SGLANG_METRICS_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics.jsonl"
    export SGLANG_METRICS_SAMPLER_LOG="${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics_sampler.log"
    export PARTIAL_ASYNC_JSONL="${LOG_ROOT}/${EXPERIMENT_NAME}_partial_async.jsonl"

    echo "===== Running ${EXPERIMENT_NAME} ====="
    set +e
    bash examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent_with_sglang_metrics.sh
    status=$?
    set -e

    echo "===== Plotting ${EXPERIMENT_NAME} ====="
    if [[ -s "${SGLANG_METRICS_FILE}" ]]; then
        set +e
        python3 examples/agent_loop/plot_sglang_running_requests.py \
            --jsonl "${SGLANG_METRICS_FILE}" \
            --title "Qwen3-8B partial_async SGLang running requests, sleep=${sleep_ms}ms, min_tool_calls=${GSM8K_MIN_TOOL_CALLS}" \
            >"${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log" 2>&1
        active_plot_status=$?

        python3 examples/agent_loop/plot_sglang_step_zoom.py \
            --jsonl "${SGLANG_METRICS_FILE}" \
            --output-png "${LOG_ROOT}/${EXPERIMENT_NAME}_two_main_peaks_zoom_0p1s.png" \
            --title "Qwen3-8B partial_async running requests zoom, sleep=${sleep_ms}ms" \
            --steps "${ZOOM_STEPS:-1,2}" \
            >"${LOG_ROOT}/${EXPERIMENT_NAME}_plot_two_main_peaks_zoom.log" 2>&1
        zoom_plot_status=$?
        set -e

        if [[ "${active_plot_status}" != "0" ]]; then
            echo "Active-window plot failed for ${EXPERIMENT_NAME}; see ${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log"
        else
            cat "${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log"
        fi

        if [[ "${zoom_plot_status}" != "0" ]]; then
            echo "Zoom plot failed for ${EXPERIMENT_NAME}; see ${LOG_ROOT}/${EXPERIMENT_NAME}_plot_two_main_peaks_zoom.log"
        else
            cat "${LOG_ROOT}/${EXPERIMENT_NAME}_plot_two_main_peaks_zoom.log"
        fi
    else
        echo "No metrics file found for ${EXPERIMENT_NAME}: ${SGLANG_METRICS_FILE}"
    fi

    if [[ -s "${PARTIAL_ASYNC_JSONL}" ]]; then
        echo "partial_async_jsonl=${PARTIAL_ASYNC_JSONL}"
    else
        echo "No partial_async JSONL found for ${EXPERIMENT_NAME}: ${PARTIAL_ASYNC_JSONL}"
    fi

    if [[ "${status}" != "0" ]]; then
        echo "Run ${EXPERIMENT_NAME} exited with status ${status}; continuing to next sleep."
    fi

    if command -v ray >/dev/null 2>&1; then
        ray stop --force || true
        sleep 5
    fi
done

echo "Finished Qwen3-8B partial_async grid: ${RUN_ID}"
