#!/usr/bin/env bash
# Run GSM8K tool-agent latency profiling with exactly 2 minimum tool calls.
#
# Default grid:
#   GSM8K_TOOL_SLEEP_MS in 200 300 400 500 600 700
#   GSM8K_MIN_TOOL_CALLS = 2
#
# Example:
#   cd /root/verl
#   export CUDA_VISIBLE_DEVICES=2,3
#   export NGPUS_PER_NODE=2
#   bash examples/agent_loop/run_gsm8k_tool_latency_2calls_200_700.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export SLEEP_MS_GRID=${SLEEP_MS_GRID:-"200 300 400 500 600 700"}
export TOOL_CALL_GRID=${TOOL_CALL_GRID:-"2"}
export GRID_EXPERIMENT_PREFIX=${GRID_EXPERIMENT_PREFIX:-qwen3_1.7b_tool_latency_2calls}

bash "${SCRIPT_DIR}/run_gsm8k_tool_latency_grid.sh" "$@"
