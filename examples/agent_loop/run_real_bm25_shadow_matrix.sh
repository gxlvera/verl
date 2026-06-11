#!/usr/bin/env bash
# Real-BM25 shadow matrix. 20 runs total, for bs in {256,384}:
#   (A) 14 shadow-only: baseline + (Full/Tail25/Tail15) x (await/fire-and-forget).
#   (B)  6 spec-hit (await only): Full (per-turn 30/30/18/15) + Tail25/Tail15 (18%).
# Each run logs shadow decode lengths via HOTPOT_SHADOW_JSONL.
#
# A single failed run is logged and skipped (no set -e on the run loop) so the
# whole matrix still completes.
set -uo pipefail
cd "$(dirname "$0")/../.."

RETRIEVER_PY=${RETRIEVER_PY:-/home/tiger/retriever_env/bin/python}
BM25_INDEX_DIR=${BM25_INDEX_DIR:-/home/tiger/data/search_r1_wiki18/bm25s_index}
SEARCH_HOST=127.0.0.1
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_TOPK=${SEARCH_TOPK:-3}
SERVER_LOG=/home/tiger/verl/logs/bm25_server.log
TS=$(date +%Y%m%d_%H%M%S)
echo "$TS" > /tmp/real_bm25_matrix_ts.txt
SUMMARY=/home/tiger/verl/logs/real_bm25_matrix_${TS}_summary.txt
echo "matrix ts=$TS  (summary -> $SUMMARY)"

# ---------- boot bm25 server (once) ----------
if ! curl -sf "http://${SEARCH_HOST}:${SEARCH_PORT}/health" >/dev/null 2>&1; then
    echo "[bm25] starting server ..."
    LOCAL_BM25_INDEX_DIR="${BM25_INDEX_DIR}" ONLINE_SEARCH_TOPK="${SEARCH_TOPK}" \
        nohup "${RETRIEVER_PY}" examples/agent_loop/online_search_server.py \
        --provider localbm25 --host "${SEARCH_HOST}" --port "${SEARCH_PORT}" \
        --topk "${SEARCH_TOPK}" --bm25-index-dir "${BM25_INDEX_DIR}" \
        > "${SERVER_LOG}" 2>&1 &
    echo "[bm25] pid $! ; waiting for /health ..."
    ok=0
    for i in $(seq 1 240); do
        if curl -sf "http://${SEARCH_HOST}:${SEARCH_PORT}/health" >/dev/null 2>&1; then
            echo "[bm25] healthy after ${i}s"; ok=1; break
        fi
        sleep 1
    done
    if [ "$ok" != "1" ]; then echo "[bm25] FAILED to start"; tail -n 30 "$SERVER_LOG"; exit 1; fi
else
    echo "[bm25] server already healthy"
fi

# ---------- common experiment env ----------
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
    local bs="$1" name="$2"; shift 2
    local tag="real_${name}_bs${bs}_${TS}"
    echo "===== RUN ${tag} ====="
    env "${COMMON[@]}" "$@" \
        BS_GRID="${bs}" \
        HOTPOT_SHADOW_JSONL=/home/tiger/verl/logs/sj_${tag} \
        SWEEP_TAG="${tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh \
        > /home/tiger/verl/logs/${tag}_driver.log 2>&1
    local L
    L=$(ls logs/baseline_*${tag}_bs${bs}.log 2>/dev/null | grep -v -e plot_active -e sampler -e metrics | head -1)
    local tp="NA"
    [ -n "$L" ] && tp=$(grep -oE "perf/throughput:[0-9.]+" "$L" | tail -1)
    echo "${tag} -> ${tp:-FAILED}" | tee -a "$SUMMARY"
}

for bs in 256 384; do
    run_one "$bs" baseline
    run_one "$bs" full_await    HOTPOT_SHADOW_NONTHINKING=true HOTPOT_SHADOW_FIRE_AND_FORGET=false
    run_one "$bs" tail25_await  HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.25 HOTPOT_SHADOW_FIRE_AND_FORGET=false
    run_one "$bs" tail15_await  HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.15 HOTPOT_SHADOW_FIRE_AND_FORGET=false
    run_one "$bs" full_faf      HOTPOT_SHADOW_NONTHINKING=true HOTPOT_SHADOW_FIRE_AND_FORGET=true
    run_one "$bs" tail25_faf    HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.25 HOTPOT_SHADOW_FIRE_AND_FORGET=true
    run_one "$bs" tail15_faf    HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.15 HOTPOT_SHADOW_FIRE_AND_FORGET=true
done

# ---------- spec-hit experiments (await only; per-turn hit rates) ----------
# Full shadow: per-turn 30/30/18/15 ; Tail shadow: uniform 18% per turn.
for bs in 256 384; do
    run_one "$bs" full_spec_await   HOTPOT_SHADOW_NONTHINKING=true HOTPOT_SHADOW_FIRE_AND_FORGET=false HOTPOT_SPEC_HIT_RATE_BY_TURN=0.30,0.30,0.18,0.15
    run_one "$bs" tail25_spec_await HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.25 HOTPOT_SHADOW_FIRE_AND_FORGET=false HOTPOT_SPEC_HIT_RATE_BY_TURN=0.18
    run_one "$bs" tail15_spec_await HOTPOT_SHADOW_TAIL=true HOTPOT_SHADOW_TAIL_ACTIVE_FRAC=0.15 HOTPOT_SHADOW_FIRE_AND_FORGET=false HOTPOT_SPEC_HIT_RATE_BY_TURN=0.18
done

echo "ALL_DONE ts=$TS"
