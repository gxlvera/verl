#!/usr/bin/env bash
# Qwen3-8B, bs256, real BM25 (NO simulated tool latency). Full shadow every turn,
# await, no spec. DIRECT-TOKEN shadow: the shadow is fed the SAME accumulated
# token_ids as the main thinking request (NO apply_chat_template, NO 1024 prompt
# truncation), with an empty-think prefix appended to make it non-thinking.
#
# Two runs:
#   A) direct_nocompress : shadow sees the full main context (incl. prior <think>).
#   B) direct_compress   : prior assistant turns' <think>...</think> token spans
#                          stripped (sys-prompt <think> text preserved).
#
# Both dump ALL traces (main prompt + shadow prompt recorded separately, main &
# shadow outputs) and record long-tail checkpoints (active samples down to
# 25/15/10/5% of peak -> count tool calls fired in the remaining tail window).
set -uo pipefail
cd "$(dirname "$0")/../.."

SEARCH_HOST=127.0.0.1
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_TOPK=${SEARCH_TOPK:-3}
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_direct_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_direct_${TS}_summary.txt
echo "direct-set ts=$TS  (summary -> $SUMMARY)"

if ! curl -sf "http://${SEARCH_HOST}:${SEARCH_PORT}/health" >/dev/null 2>&1; then
    echo "[bm25] server NOT healthy on :${SEARCH_PORT} -- start it first"; exit 1
fi
echo "[bm25] server healthy"

COMMON=(
  DATA_DIR=/home/tiger/data/hotpotqa_search_r1_react
  TOOL_FILE=examples/agent_loop/hotpot_online_search_tool.py
  HOTPOT_AUTO_TOOL_NAME=retrieve_hotpot_context
  ONLINE_SEARCH_RETRIEVAL_URL=http://${SEARCH_HOST}:${SEARCH_PORT}/retrieve
  ONLINE_SEARCH_TOPK=${SEARCH_TOPK}
  ONLINE_SEARCH_TIMEOUT=30
  ONLINE_SEARCH_MAX_CHARS=4000
  MAX_MODEL_LEN=8192 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=7168
  PER_TURN_MAX_RESPONSE_LENGTH=500
  HOTPOT_MIN_TOOL_CALLS=1 HOTPOT_MAX_TOOL_CALLS=4
  DATA_ENABLE_THINKING=True HOTPOT_MAIN_ENABLE_THINKING=true
  HOTPOT_SHADOW_TAIL_MIN_TURN=3
  HOTPOT_SPEC_HIT_RATE=0
  HOTPOT_SPEC_HIT_SEED=7
  LOG_ROOT=/home/tiger/verl/logs QUEUE_TIME_DIR=/home/tiger/verl/logs
)

run_one() {
    local name="$1"; shift
    local tag="dir_${name}_bs256_${TS}"
    echo "===== RUN ${tag} ====="
    env "${COMMON[@]}" "$@" \
        BS_GRID=256 \
        HOTPOT_SHADOW_JSONL=/home/tiger/verl/logs/sj_${tag} \
        HOTPOT_TRACE_DUMP=/home/tiger/verl/logs/trace_${tag} \
        HOTPOT_TRACE_DUMP_TURNS=all \
        HOTPOT_TRACE_DUMP_MAX=8000 \
        HOTPOT_TAIL_CKPT=/home/tiger/verl/logs/tailckpt_${tag} \
        SWEEP_TAG="${tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh \
        > /home/tiger/verl/logs/${tag}_driver.log 2>&1
    local L tp="FAILED"
    L=$(ls logs/baseline_*${tag}_bs256.log 2>/dev/null | grep -v -e plot_active -e sampler -e metrics | head -1)
    [ -n "$L" ] && tp=$(grep -oE "perf/throughput:[0-9.]+" "$L" | tail -1)
    echo "${tag} -> ${tp:-FAILED}  trace=/home/tiger/verl/logs/trace_${tag}  tailckpt=/home/tiger/verl/logs/tailckpt_${tag}" | tee -a "$SUMMARY"
}

# A) non-compress, direct-token shadow (full main context incl. thinking trace).
run_one direct_nocompress \
    HOTPOT_SHADOW_NONTHINKING=true \
    HOTPOT_SHADOW_FIRE_AND_FORGET=false \
    HOTPOT_SHADOW_DIRECT_TOKENS=true

# B) compress, direct-token shadow (prior <think>...</think> stripped at token level).
run_one direct_compress \
    HOTPOT_SHADOW_NONTHINKING=true \
    HOTPOT_SHADOW_FIRE_AND_FORGET=false \
    HOTPOT_SHADOW_DIRECT_TOKENS=true \
    HOTPOT_SHADOW_COMPRESS=true

echo "DIRECT_DONE ts=$TS"
