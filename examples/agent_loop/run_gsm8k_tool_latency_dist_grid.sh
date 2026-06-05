#!/usr/bin/env bash
# Run a fixed grid of weighted random GSM8K tool latency distributions.
#
# Distributions:
#   300:30,600:70
#   350:30,600:70
#   400:30,600:70
#   0:40,300:30,600:30
#   0:60,300:30,600:10
#   0:80,600:20
#
# Example:
#   cd /root/verl
#   export CUDA_VISIBLE_DEVICES=2,3
#   export NGPUS_PER_NODE=2
#   GSM8K_TOOL_SLEEP_SEED=42 bash examples/agent_loop/run_gsm8k_tool_latency_dist_grid.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

DIST_GRID=${DIST_GRID:-"300:30,600:70;350:30,600:70;400:30,600:70;0:40,300:30,600:30;0:60,300:30,600:10;0:80,600:20"}
GRID_STOP_RAY_BETWEEN_RUNS=${GRID_STOP_RAY_BETWEEN_RUNS:-1}
GRID_RUN_ID=${GRID_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
GRID_EXPERIMENT_PREFIX=${GRID_EXPERIMENT_PREFIX:-qwen3_1.7b_tool_latency_dist_grid}

export PROJECT_NAME=${PROJECT_NAME:-specTool}
export LOG_ROOT=${LOG_ROOT:-/root/logs}
export LOG_TO_FILE=${LOG_TO_FILE:-1}
export GSM8K_TOOL_SLEEP_SEED=${GSM8K_TOOL_SLEEP_SEED:-42}
export GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-2}

mkdir -p "${LOG_ROOT}"
GRID_LOG_FILE=${GRID_LOG_FILE:-${LOG_ROOT}/${GRID_EXPERIMENT_PREFIX}_${GRID_RUN_ID}.log}
exec > >(tee -a "${GRID_LOG_FILE}") 2>&1

dist_tag() {
    printf '%s' "$1" | tr ',:' '__' | tr -cd '[:alnum:]_'
}

run_one() {
    local dist="$1"
    local tag
    tag="$(dist_tag "${dist}")"
    local exp_name="${GRID_EXPERIMENT_PREFIX}_${tag}_seed${GSM8K_TOOL_SLEEP_SEED}_toolcalls${GSM8K_MIN_TOOL_CALLS}_${GRID_RUN_ID}"

    echo
    echo "================================================================"
    echo "Starting experiment: ${exp_name}"
    echo "  GSM8K_TOOL_SLEEP_DIST=${dist}"
    echo "  GSM8K_TOOL_SLEEP_SEED=${GSM8K_TOOL_SLEEP_SEED}"
    echo "  GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS}"
    echo "================================================================"

    if [[ "${GRID_STOP_RAY_BETWEEN_RUNS}" == "1" ]]; then
        ray stop -f || true
    fi

    export GSM8K_TOOL_SLEEP_DIST="${dist}"
    export EXPERIMENT_NAME="${exp_name}"

    bash "${SCRIPT_DIR}/run_gsm8k_tool_latency_dist.sh"

    if [[ "${GRID_STOP_RAY_BETWEEN_RUNS}" == "1" ]]; then
        ray stop -f || true
    fi

    echo "Finished experiment: ${exp_name}"
}

echo "Grid run id: ${GRID_RUN_ID}"
echo "Grid log: ${GRID_LOG_FILE}"
echo "Repo root: ${REPO_ROOT}"
echo "Distribution grid: ${DIST_GRID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NGPUS_PER_NODE=${NGPUS_PER_NODE:-<unset>}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-<base default>}"
echo "ROLLOUT_N=${ROLLOUT_N:-<base default>}"
echo "TOTAL_STEPS=${TOTAL_STEPS:-<base default>}"
echo "GSM8K_TOOL_SLEEP_SEED=${GSM8K_TOOL_SLEEP_SEED}"
echo "GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS}"
echo "GRID_STOP_RAY_BETWEEN_RUNS=${GRID_STOP_RAY_BETWEEN_RUNS}"

IFS=';' read -ra dists <<< "${DIST_GRID}"
for dist in "${dists[@]}"; do
    run_one "${dist}"
done

echo
echo "All distribution-grid experiments finished."
