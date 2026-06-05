#!/usr/bin/env bash
# Run GSM8K tool-agent profiling with a weighted random tool latency distribution.
#
# Example:
#   cd /root/verl
#   export CUDA_VISIBLE_DEVICES=2,3
#   export NGPUS_PER_NODE=2
#   GSM8K_TOOL_SLEEP_DIST="0:50,600:50" GSM8K_TOOL_SLEEP_SEED=42 \
#     bash examples/agent_loop/run_gsm8k_tool_latency_dist.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export GSM8K_TOOL_SLEEP_MS=${GSM8K_TOOL_SLEEP_MS:-0}
export GSM8K_TOOL_SLEEP_DIST=${GSM8K_TOOL_SLEEP_DIST:?Set GSM8K_TOOL_SLEEP_DIST, for example '0:50,600:50'.}
export GSM8K_TOOL_SLEEP_SEED=${GSM8K_TOOL_SLEEP_SEED:-42}
export GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-2}

dist_tag="$(printf '%s' "${GSM8K_TOOL_SLEEP_DIST}" | tr ',:' '__' | tr -cd '[:alnum:]_')"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_tool_latency_dist_${dist_tag}_seed${GSM8K_TOOL_SLEEP_SEED}_toolcalls${GSM8K_MIN_TOOL_CALLS}_$(date +%Y%m%d_%H%M%S)}

bash "${SCRIPT_DIR}/run_qwen3_1.7b_gsm8k_tool_agent.sh" "$@"
