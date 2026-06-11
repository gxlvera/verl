#!/usr/bin/env bash
# bs256, BM25 + SerpAPI latency, await. Full shadow (no spec), but SHADOW decode
# capped at 150 new tokens (HOTPOT_NON_THINKING_MAX_NEW_TOKENS=150) to test whether
# limiting tail-shadow ramble recovers throughput vs full_await (which was -34%).
set -uo pipefail
cd "$(dirname "$0")/../.."

LATENCY_CSV=/home/tiger/verl/examples/agent_loop/results/qwen3_8b_hotpotqa_serpapi_latency_100_no_thinking/search_latencies.csv
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_cap150_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_cap150_${TS}_summary.txt
echo "cap150-set ts=$TS  (summary -> $SUMMARY)"

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

# Full shadow every turn, await, no spec, shadow decode capped at 150 tokens.
run_one full_await_cap150 \
    HOTPOT_SHADOW_NONTHINKING=true \
    HOTPOT_SHADOW_FIRE_AND_FORGET=false \
    HOTPOT_NON_THINKING_MAX_NEW_TOKENS=150

echo "CAP150_DONE ts=$TS"
