#!/usr/bin/env bash
# Qwen3-8B, bs256, real BM25 (NO simulated tool latency). Full shadow every turn,
# await (so the shadow always completes and can be paired with the main output),
# no spec. NEW: HOTPOT_SHADOW_COMPRESS=true -> the non-thinking shadow prompt is
# rebuilt as  sys + user + (assistant tool_call WITHOUT thinking) + tool_response
# (alternating), i.e. it never sees the main model's prior <think> traces. Tests
# whether stripping the thinking traces stops the shadow from rambling in later
# turns. All traces dumped: main prompt, shadow prompt, main output, shadow output.
set -uo pipefail
cd "$(dirname "$0")/../.."

SEARCH_HOST=127.0.0.1
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_TOPK=${SEARCH_TOPK:-3}
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_compress_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_compress_${TS}_summary.txt
echo "compress-set ts=$TS  (summary -> $SUMMARY)"

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
    local tag="cmp_${name}_bs256_${TS}"
    echo "===== RUN ${tag} ====="
    env "${COMMON[@]}" "$@" \
        BS_GRID=256 \
        HOTPOT_SHADOW_JSONL=/home/tiger/verl/logs/sj_${tag} \
        HOTPOT_TRACE_DUMP=/home/tiger/verl/logs/trace_${tag} \
        HOTPOT_TRACE_DUMP_TURNS=all \
        HOTPOT_TRACE_DUMP_MAX=8000 \
        SWEEP_TAG="${tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh \
        > /home/tiger/verl/logs/${tag}_driver.log 2>&1
    local L tp="FAILED"
    L=$(ls logs/baseline_*${tag}_bs256.log 2>/dev/null | grep -v -e plot_active -e sampler -e metrics | head -1)
    [ -n "$L" ] && tp=$(grep -oE "perf/throughput:[0-9.]+" "$L" | tail -1)
    echo "${tag} -> ${tp:-FAILED}  trace=/home/tiger/verl/logs/trace_${tag}" | tee -a "$SUMMARY"
}

# Full shadow every turn, await, no spec, COMPRESSED shadow prompt (no thinking trace).
run_one full_await_compress \
    HOTPOT_SHADOW_NONTHINKING=true \
    HOTPOT_SHADOW_FIRE_AND_FORGET=false \
    HOTPOT_SHADOW_COMPRESS=true

echo "COMPRESS_DONE ts=$TS"
