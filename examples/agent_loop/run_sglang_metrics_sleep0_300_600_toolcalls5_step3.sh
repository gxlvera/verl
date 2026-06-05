#!/usr/bin/env bash
# Run sleep=0/300/600ms experiments with at least 5 tool calls, then plot active-window and step2/step3 zoom charts.

set -euo pipefail

LOG_ROOT=${LOG_ROOT:-/root/logs}
mkdir -p "${LOG_ROOT}"

SLEEP_MS_GRID=${SLEEP_MS_GRID:-"0 300 600"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

export GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-5}
export SGLANG_METRICS_INTERVAL=${SGLANG_METRICS_INTERVAL:-0.1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export ZOOM_STEPS=${ZOOM_STEPS:-2,3}

echo "Running SGLang metrics grid: sleeps=${SLEEP_MS_GRID}, min_tool_calls=${GSM8K_MIN_TOOL_CALLS}, run_id=${RUN_ID}"
echo "Common config: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}, PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}, PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU}, ROLLOUT_N=${ROLLOUT_N}, TOTAL_STEPS=${TOTAL_STEPS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, NGPUS_PER_NODE=${NGPUS_PER_NODE}, ZOOM_STEPS=${ZOOM_STEPS}"

for sleep_ms in ${SLEEP_MS_GRID}; do
    export GSM8K_TOOL_SLEEP_MS="${sleep_ms}"
    export EXPERIMENT_NAME="sglang_metrics_sleep${sleep_ms}_toolcalls${GSM8K_MIN_TOOL_CALLS}_bs${TRAIN_BATCH_SIZE}_rollout${ROLLOUT_N}_step${TOTAL_STEPS}_${RUN_ID}"
    export RUN_LOG_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}.log"
    export SGLANG_METRICS_FILE="${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics.jsonl"
    export SGLANG_METRICS_SAMPLER_LOG="${LOG_ROOT}/${EXPERIMENT_NAME}_sglang_metrics_sampler.log"

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
            --title "SGLang running requests, sleep=${sleep_ms}ms, min_tool_calls=${GSM8K_MIN_TOOL_CALLS}, bs=${TRAIN_BATCH_SIZE}, rollout_n=${ROLLOUT_N}" \
            >"${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log" 2>&1
        active_plot_status=$?

        python3 examples/agent_loop/plot_sglang_step_zoom.py \
            --jsonl "${SGLANG_METRICS_FILE}" \
            --output-png "${LOG_ROOT}/${EXPERIMENT_NAME}_step2_3_zoom_0p1s.png" \
            --title "SGLang running requests zoom, sleep=${sleep_ms}ms, min_tool_calls=${GSM8K_MIN_TOOL_CALLS}, selected peaks" \
            --steps "${ZOOM_STEPS}" \
            >"${LOG_ROOT}/${EXPERIMENT_NAME}_plot_step2_3_zoom.log" 2>&1
        zoom_plot_status=$?
        set -e

        if [[ "${active_plot_status}" != "0" ]]; then
            echo "Active-window plot failed for ${EXPERIMENT_NAME}; see ${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log"
        else
            cat "${LOG_ROOT}/${EXPERIMENT_NAME}_plot_active.log"
        fi

        if [[ "${zoom_plot_status}" != "0" ]]; then
            echo "Step2/step3 zoom plot failed for ${EXPERIMENT_NAME}; see ${LOG_ROOT}/${EXPERIMENT_NAME}_plot_step2_3_zoom.log"
        else
            cat "${LOG_ROOT}/${EXPERIMENT_NAME}_plot_step2_3_zoom.log"
        fi
    else
        echo "No metrics file found for ${EXPERIMENT_NAME}: ${SGLANG_METRICS_FILE}"
    fi

    if [[ "${status}" != "0" ]]; then
        echo "Run ${EXPERIMENT_NAME} exited with status ${status}; continuing to next sleep."
    fi

    if command -v ray >/dev/null 2>&1; then
        ray stop --force || true
        sleep 5
    fi
done

echo "Finished SGLang metrics grid: ${RUN_ID}"
