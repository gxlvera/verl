#!/usr/bin/env bash
# BM25 (real content) + simulated SerpAPI latency (mean ~2.7s). bs256, ALL await.
# Tail-fraction sweep: 15/10/5%. Runs:
#   tail15_await, tail10_await, tail10_spec_await, tail5_await, tail5_spec_await.
# Tail spec uniform hit rate 0.18. A failed run is logged and skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."

SEARCH_HOST=127.0.0.1
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_TOPK=${SEARCH_TOPK:-3}
LATENCY_CSV=/home/tiger/verl/examples/agent_loop/results/qwen3_8b_hotpotqa_serpapi_latency_100_no_thinking/search_latencies.csv
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_tailsweep_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_tailsweep_${TS}_summary.txt
echo "tailsweep-set ts=$TS  (summary -> $SUMMARY)"

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
    local bs="$1" name="$2"; shift 2
    local tag="lat_${name}_bs${bs}_${TS}"
    echo "===== RUN ${tag} ====="
    env "${COMMON[@]}" "$@" \
        BS_GRID="${bs}" \
        HOTPOT_SHADOW_JSONL=/home/tiger/verl/logs/sj_${tag} \
        SWEEP_TAG="${tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh \
        > /home/tiger/verl/logs/${tag}_driver.log 2>&1
    local L
    L=$(ls logs/baseline_*${tag}_bs${bs}.log 2>/dev/null | grep -v -e plot_active -e sampler -e metrics | head -1)
    local tp="NA" ms="NA"
    if [ -n "$L" ]; then
        tp=$(grep -oE "perf/throughput:[0-9.]+" "$L" | tail -1)
        ms=$(grep -oE "perf/time_per_step:[0-9.]+" "$L" | tail -1)
    fi
    echo "${tag} -> ${tp:-FAILED} ${ms}" | tee -a "$SUMMARY"
}

bs=256
run_one "$bs" tail15_await      HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.15 HOTPOT_SHADOW_FIRE_AND_FORGET=false
run_one "$bs" tail10_await      HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.10 HOTPOT_SHADOW_FIRE_AND_FORGET=false
run_one "$bs" tail10_spec_await HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.10 HOTPOT_SHADOW_FIRE_AND_FORGET=false HOTPOT_SPEC_HIT_RATE_BY_TURN=0.18
run_one "$bs" tail5_await       HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.05 HOTPOT_SHADOW_FIRE_AND_FORGET=false
run_one "$bs" tail5_spec_await  HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.05 HOTPOT_SHADOW_FIRE_AND_FORGET=false HOTPOT_SPEC_HIT_RATE_BY_TURN=0.18

echo "ALL_DONE ts=$TS"
