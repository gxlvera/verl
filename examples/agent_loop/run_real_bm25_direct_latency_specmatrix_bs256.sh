#!/usr/bin/env bash
# Qwen3-8B, bs256, real BM25 + simulated SerpAPI latency (mean ~2.7s). DIRECT-TOKEN
# shadow (bug-fixed: shadow reuses main's untruncated token_ids + empty-think
# suffix; no apply_chat_template, no 1024 truncation). All await.
#
# Runs (vs baseline):
#   shadow-only (no spec): full / tail25 / tail15 / tail10 / tail5
#   spec (await, per-turn hit rates measured from run A):
#       full / tail25 / tail15 / tail10 / tail5
# Per-turn spec hit rates (name+args match observed in the fixed full-shadow run):
#   t1=0.283 t2=0.696 t3=0.722 t4=0.834 t5=0.909  (tail shadows only use t>=3).
set -uo pipefail
cd "$(dirname "$0")/../.."

SEARCH_HOST=127.0.0.1
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_TOPK=${SEARCH_TOPK:-3}
LATENCY_CSV=/home/tiger/verl/examples/agent_loop/results/qwen3_8b_hotpotqa_serpapi_latency_100_no_thinking/search_latencies.csv
SPEC_RATES=0.283,0.696,0.722,0.834,0.909
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_direct_lat_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_direct_lat_${TS}_summary.txt
echo "direct-lat-set ts=$TS  (summary -> $SUMMARY)"

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
  HOTPOT_TOOL_LATENCY_CSV=${LATENCY_CSV}
  HOTPOT_TOOL_LATENCY_CSV_COLUMN=search_elapsed_s
  HOTPOT_TOOL_SLEEP_SEED=7
  LOG_ROOT=/home/tiger/verl/logs QUEUE_TIME_DIR=/home/tiger/verl/logs
)

run_one() {
    local name="$1"; shift
    local tag="dl_${name}_bs256_${TS}"
    echo "===== RUN ${tag} ====="
    env "${COMMON[@]}" "$@" \
        BS_GRID=256 \
        HOTPOT_SHADOW_JSONL=/home/tiger/verl/logs/sj_${tag} \
        SWEEP_TAG="${tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh \
        > /home/tiger/verl/logs/${tag}_driver.log 2>&1
    local L tp="FAILED" ms="NA"
    L=$(ls logs/baseline_*${tag}_bs256.log 2>/dev/null | grep -v -e plot_active -e sampler -e metrics | head -1)
    if [ -n "$L" ]; then
        tp=$(grep -oE "perf/throughput:[0-9.]+" "$L" | tail -1)
        ms=$(grep -oE "perf/time_per_step:[0-9.]+" "$L" | tail -1)
    fi
    echo "${tag} -> ${tp:-FAILED} ${ms}" | tee -a "$SUMMARY"
}

DIRECT=(HOTPOT_SHADOW_DIRECT_TOKENS=true HOTPOT_SHADOW_FIRE_AND_FORGET=false)

# baseline (no shadow, no spec)
run_one baseline

# shadow-only, no spec
run_one full_await    "${DIRECT[@]}" HOTPOT_SHADOW_NONTHINKING=true
run_one tail25_await  "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.25
run_one tail15_await  "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.15
run_one tail10_await  "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.10
run_one tail5_await   "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.05

# spec (await, per-turn measured hit rates)
run_one full_spec_await    "${DIRECT[@]}" HOTPOT_SHADOW_NONTHINKING=true HOTPOT_SPEC_HIT_RATE_BY_TURN=${SPEC_RATES}
run_one tail25_spec_await  "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.25 HOTPOT_SPEC_HIT_RATE_BY_TURN=${SPEC_RATES}
run_one tail15_spec_await  "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.15 HOTPOT_SPEC_HIT_RATE_BY_TURN=${SPEC_RATES}
run_one tail10_spec_await  "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.10 HOTPOT_SPEC_HIT_RATE_BY_TURN=${SPEC_RATES}
run_one tail5_spec_await   "${DIRECT[@]}" HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.05 HOTPOT_SPEC_HIT_RATE_BY_TURN=${SPEC_RATES}

echo "DIRECT_LAT_DONE ts=$TS"
