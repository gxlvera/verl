#!/usr/bin/env bash
# Run a latency x tool-call-count grid for GSM8K tool-agent rollout profiling.
#
# Default grid:
#   GSM8K_TOOL_SLEEP_MS in 600 500 400 300 200 100
#   GSM8K_MIN_TOOL_CALLS in 4 3 2 1
#
# Example:
#   cd /root/verl
#   tmux new -s tool-grid
#   export CUDA_VISIBLE_DEVICES=4,5
#   export NGPUS_PER_NODE=2
#   bash examples/agent_loop/run_gsm8k_tool_latency_grid.sh
#
# Useful overrides:
#   TOTAL_STEPS=10 bash examples/agent_loop/run_gsm8k_tool_latency_grid.sh
#   SLEEP_MS_GRID="100 300 600" TOOL_CALL_GRID="1 2" bash examples/agent_loop/run_gsm8k_tool_latency_grid.sh
#   GRID_STOP_RAY_BETWEEN_RUNS=0 bash examples/agent_loop/run_gsm8k_tool_latency_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_qwen3_1.7b_gsm8k_tool_agent.sh"

cd "${REPO_ROOT}"

SLEEP_MS_GRID=${SLEEP_MS_GRID:-"600 500 400 300 200 100"}
TOOL_CALL_GRID=${TOOL_CALL_GRID:-"4 3 2 1"}
GRID_STOP_RAY_BETWEEN_RUNS=${GRID_STOP_RAY_BETWEEN_RUNS:-1}
GRID_RUN_ID=${GRID_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
GRID_EXPERIMENT_PREFIX=${GRID_EXPERIMENT_PREFIX:-qwen3_1.7b_tool_latency_grid}

export PROJECT_NAME=${PROJECT_NAME:-specTool}
export LOG_ROOT=${LOG_ROOT:-/root/logs}
export LOG_TO_FILE=${LOG_TO_FILE:-1}

mkdir -p "${LOG_ROOT}"
GRID_LOG_FILE=${GRID_LOG_FILE:-${LOG_ROOT}/${GRID_EXPERIMENT_PREFIX}_${GRID_RUN_ID}.log}
exec > >(tee -a "${GRID_LOG_FILE}") 2>&1

echo "Grid run id: ${GRID_RUN_ID}"
echo "Grid log: ${GRID_LOG_FILE}"
echo "Repo root: ${REPO_ROOT}"
echo "Sleep grid ms: ${SLEEP_MS_GRID}"
echo "Tool-call grid: ${TOOL_CALL_GRID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE:-<unset>}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-<base default>}"
echo "ROLLOUT_N=${ROLLOUT_N:-<base default>}"
echo "TOTAL_STEPS=${TOTAL_STEPS:-<base default>}"
echo "GRID_STOP_RAY_BETWEEN_RUNS=${GRID_STOP_RAY_BETWEEN_RUNS}"

run_one() {
    local sleep_ms="$1"
    local min_tool_calls="$2"
    local exp_name="${GRID_EXPERIMENT_PREFIX}_sleep${sleep_ms}ms_toolcalls${min_tool_calls}_${GRID_RUN_ID}"

    echo
    echo "================================================================"
    echo "Starting experiment: ${exp_name}"
    echo "  GSM8K_TOOL_SLEEP_MS=${sleep_ms}"
    echo "  GSM8K_MIN_TOOL_CALLS=${min_tool_calls}"
    echo "================================================================"

    if [[ "${GRID_STOP_RAY_BETWEEN_RUNS}" == "1" ]]; then
        ray stop -f || true
    fi

    export GSM8K_TOOL_SLEEP_MS="${sleep_ms}"
    export GSM8K_TOOL_SLEEP_DIST=""
    export GSM8K_MIN_TOOL_CALLS="${min_tool_calls}"
    export EXPERIMENT_NAME="${exp_name}"

    bash "${BASE_SCRIPT}"

    if [[ "${GRID_STOP_RAY_BETWEEN_RUNS}" == "1" ]]; then
        ray stop -f || true
    fi

    echo "Finished experiment: ${exp_name}"
}

for sleep_ms in ${SLEEP_MS_GRID}; do
    for min_tool_calls in ${TOOL_CALL_GRID}; do
        run_one "${sleep_ms}" "${min_tool_calls}"
    done
done

echo
echo "All grid experiments finished."
