#!/usr/bin/env bash
# Extra make-up runs: bs256, BM25 + SerpAPI latency, await.
#   baseline_extra (makeup for failed r2) + tail5_await_extra (repeat confirm).
set -uo pipefail
cd "$(dirname "$0")/../.."

LATENCY_CSV=/home/tiger/verl/examples/agent_loop/results/qwen3_8b_hotpotqa_serpapi_latency_100_no_thinking/search_latencies.csv
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_extra2_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_extra2_${TS}_summary.txt
echo "extra2-set ts=$TS  (summary -> $SUMMARY)"

if ! curl -sf "http://127.0.0.1:8000/health" >/dev/null 2>&1; then
    echo "[bm25] server NOT healthy -- start it first"; exit 1
fi

COMMON=(
  DATA_DIR=/home/tiger/data/hotpotqa_search_r1_react
  TOOL_FILE=examples/agent_loop/hotpot_online_search_tool.py
  HOTPOT_AUTO_TOOL_NAME=retrieve_hotpot_context
  ONLINE_SEARCH_RETRIEVAL_URL=http://127.0.0.1:8000/retrieve
  ONLINE_SEARCH_TOPK=3 ONLINE_SEARCH_TIMEOUT=30 ONLINE_SEARCH_MAX_CHARS=4000
  MAX_MODEL_LEN=8192 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=7168
  PER_TURN_MAX_RESPONSE_LENGTH=500
  HOTPOT_MIN_TOOL_CALLS=1 HOTPOT_MAX_TOOL_CALLS=4
  DATA_ENABLE_THINKING=True HOTPOT_MAIN_ENABLE_THINKING=true
  HOTPOT_SHADOW_TAIL_MIN_TURN=3 HOTPOT_SPEC_HIT_RATE=0 HOTPOT_SPEC_HIT_SEED=7
  HOTPOT_TOOL_LATENCY_CSV=${LATENCY_CSV}
  HOTPOT_TOOL_LATENCY_CSV_COLUMN=search_elapsed_s HOTPOT_TOOL_SLEEP_SEED=7
  LOG_ROOT=/home/tiger/verl/logs QUEUE_TIME_DIR=/home/tiger/verl/logs
)

run_one() {
    local name="$1"; shift
    local tag="lat_${name}_bs256_${TS}"
    echo "===== RUN ${tag} ====="
    env "${COMMON[@]}" "$@" \
        BS_GRID=256 HOTPOT_SHADOW_JSONL=/home/tiger/verl/logs/sj_${tag} SWEEP_TAG="${tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh \
        > /home/tiger/verl/logs/${tag}_driver.log 2>&1
    local L tp="FAILED" ms="NA"
    L=$(ls logs/baseline_*${tag}_bs256.log 2>/dev/null | grep -v -e plot_active -e sampler -e metrics | head -1)
    if [ -n "$L" ]; then
        tp=$(grep -oE "perf/throughput:[0-9.]+" "$L" | tail -1)
        ms=$(grep -oE "perf/time_per_step:[0-9.]+" "$L" | tail -1)
    fi
    echo "${tag} -> ${tp} ${ms}" | tee -a "$SUMMARY"
}

run_one baseline_extra
run_one tail5_await_extra HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.05 HOTPOT_SHADOW_FIRE_AND_FORGET=false
echo "EXTRA2_DONE ts=$TS"
