#!/usr/bin/env bash
# Rollout-only batch-size sweep for HotpotQA Qwen3-8B.
#
# Runs one experiment per batch size, sequentially, with training skipped
# (PROFILE_ROLLOUT_ONLY=1) and a single training step each. Intended for
# profiling how rollout throughput / SGLang running+queue requests / KV token
# usage scale with the rollout batch size.
#
# Each batch size is dispatched through the existing baseline grid runner, which
# starts the SGLang /metrics sampler, runs the rollout, plots running requests,
# and stops Ray before the next batch size.
#
# Example:
#   cd /home/tiger/verl
#   bash examples/agent_loop/run_rollout_only_bs_sweep.sh
#
#   BS_GRID="512 1024 2048" \
#   bash examples/agent_loop/run_rollout_only_bs_sweep.sh

set -euo pipefail

# Batch sizes to sweep (prompts per step, before rollout_n expansion).
BS_GRID=${BS_GRID:-"1024 2048"}

# Shared config. Override any of these from the environment.
export LOG_ROOT=${LOG_ROOT:-/home/tiger/logs}
export MODEL_PATH=${MODEL_PATH:-/home/tiger/models/Qwen3-8B}
export DATA_DIR=${DATA_DIR:-/home/tiger/data/hotpotqa_tool_agent}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_STEPS=${TOTAL_STEPS:-1}
export HOTPOT_MIN_TOOL_CALLS=${HOTPOT_MIN_TOOL_CALLS:-1}
export HOTPOT_MAX_TOOL_CALLS=${HOTPOT_MAX_TOOL_CALLS:-0}
export HOTPOT_TOOL_RESPONSE_TOKENS=${HOTPOT_TOOL_RESPONSE_TOKENS:-500}
export PER_TURN_MAX_RESPONSE_LENGTH=${PER_TURN_MAX_RESPONSE_LENGTH:-500}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
export MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-12000}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export SGLANG_METRICS_INTERVAL=${SGLANG_METRICS_INTERVAL:-0.1}

# Rollout-only: skip every training-side computation.
export PROFILE_ROLLOUT_ONLY=1
# Single sleep value and single mode for this sweep.
export SLEEP_MS_GRID=${SLEEP_MS_GRID:-"0"}
export MODE_GRID=${MODE_GRID:-"baseline"}

SWEEP_TAG=${SWEEP_TAG:-rolloutonly_bssweep_$(date +%Y%m%d_%H%M%S)}

echo "=========================================================="
echo "Rollout-only batch-size sweep"
echo "  batch sizes : ${BS_GRID}"
echo "  steps/run   : ${TOTAL_STEPS}  (training skipped)"
echo "  sleep_ms    : ${SLEEP_MS_GRID}"
echo "  rollout_n   : ${ROLLOUT_N}"
echo "  tool tokens : ${HOTPOT_TOOL_RESPONSE_TOKENS}"
echo "  model       : ${MODEL_PATH}"
echo "  log root    : ${LOG_ROOT}"
echo "  sweep tag   : ${SWEEP_TAG}"
echo "=========================================================="

for bs in ${BS_GRID}; do
    echo ""
    echo ">>>>> Starting batch_size=${bs} (rollout-only, ${TOTAL_STEPS} step) <<<<<"
    export TRAIN_BATCH_SIZE="${bs}"
    # Unique run id per batch size so logs/plots/metrics don't collide.
    export RUN_ID="${SWEEP_TAG}_bs${bs}"

    set +e
    bash examples/agent_loop/run_qwen3_8b_hotpotqa_tool_latency_partial_and_baseline_grid.sh
    status=$?
    set -e

    if [[ "${status}" != "0" ]]; then
        echo ">>>>> batch_size=${bs} exited with status ${status}; continuing to next. <<<<<"
    else
        echo ">>>>> batch_size=${bs} finished. <<<<<"
    fi

    # Belt-and-suspenders: ensure Ray is down before the next batch size.
    if command -v ray >/dev/null 2>&1; then
        ray stop --force >/dev/null 2>&1 || true
        sleep 5
    fi
done

echo ""
echo "Rollout-only batch-size sweep complete: ${SWEEP_TAG}"
echo "Per-run artifacts under ${LOG_ROOT}:"
echo "  baseline_hotpotqa_qwen3_8b_sleep0_*_${SWEEP_TAG}_bs<BS>*.log"
echo "  ..._sglang_metrics.jsonl / ..._sglang_metrics_running_reqs_active.{csv,png}"
