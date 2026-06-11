#!/usr/bin/env bash
# Run the HotpotQA tool-agent rollout with the REAL local BM25 retriever
# (wiki-18 corpus) instead of the synthetic fixed-context tool.
#
# It (1) boots the bm25s retrieval server from the dedicated venv, (2) waits for
# /health, then (3) runs the standard rollout-only batch-size sweep wired to the
# real `retrieve_hotpot_context` tool and the Search-R1 ReAct dataset.
#
# Usage:
#   BS_GRID="256" bash examples/agent_loop/run_hotpot_real_bm25_bs_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# ---- retriever server config ----
RETRIEVER_PY=${RETRIEVER_PY:-/home/tiger/retriever_env/bin/python}
BM25_INDEX_DIR=${BM25_INDEX_DIR:-/home/tiger/data/search_r1_wiki18/bm25s_index}
SEARCH_HOST=${SEARCH_HOST:-127.0.0.1}
SEARCH_PORT=${SEARCH_PORT:-8000}
SEARCH_TOPK=${ONLINE_SEARCH_TOPK:-3}
SERVER_LOG=${SERVER_LOG:-/home/tiger/verl/logs/bm25_server.log}

# ---- experiment config (real retrieval, no synthetic latency/context) ----
export DATA_DIR=${DATA_DIR:-/home/tiger/data/hotpotqa_search_r1_react}
export TOOL_FILE=${TOOL_FILE:-examples/agent_loop/hotpot_online_search_tool.py}
export HOTPOT_AUTO_TOOL_NAME=${HOTPOT_AUTO_TOOL_NAME:-retrieve_hotpot_context}
export ONLINE_SEARCH_RETRIEVAL_URL=${ONLINE_SEARCH_RETRIEVAL_URL:-http://${SEARCH_HOST}:${SEARCH_PORT}/retrieve}
export ONLINE_SEARCH_TOPK=${SEARCH_TOPK}
export ONLINE_SEARCH_TIMEOUT=${ONLINE_SEARCH_TIMEOUT:-30}
# real content -> let the tool return full passages (cap via MAX_TOOL_RESPONSE_LENGTH)
export ONLINE_SEARCH_MAX_CHARS=${ONLINE_SEARCH_MAX_CHARS:-4000}
# make sure no synthetic injection/latency leaks in
unset HOTPOT_TOOL_LATENCY_CSV || true
unset HOTPOT_TOOL_RESPONSE_TOKENS || true

mkdir -p "$(dirname "$SERVER_LOG")"

start_server() {
    if curl -sf "http://${SEARCH_HOST}:${SEARCH_PORT}/health" >/dev/null 2>&1; then
        echo "[bm25] server already up on ${SEARCH_HOST}:${SEARCH_PORT}"
        return
    fi
    echo "[bm25] starting server (index=${BM25_INDEX_DIR}) ..."
    LOCAL_BM25_INDEX_DIR="${BM25_INDEX_DIR}" ONLINE_SEARCH_TOPK="${SEARCH_TOPK}" \
        nohup "${RETRIEVER_PY}" examples/agent_loop/online_search_server.py \
        --provider localbm25 --host "${SEARCH_HOST}" --port "${SEARCH_PORT}" \
        --topk "${SEARCH_TOPK}" --bm25-index-dir "${BM25_INDEX_DIR}" \
        > "${SERVER_LOG}" 2>&1 &
    echo "[bm25] server pid $! (log: ${SERVER_LOG})"
    echo "[bm25] waiting for /health (index mmap load can take ~30-60s) ..."
    for i in $(seq 1 120); do
        if curl -sf "http://${SEARCH_HOST}:${SEARCH_PORT}/health" >/dev/null 2>&1; then
            echo "[bm25] healthy after ${i}s"
            return
        fi
        sleep 1
    done
    echo "[bm25] ERROR: server did not become healthy; see ${SERVER_LOG}" >&2
    tail -n 20 "${SERVER_LOG}" >&2 || true
    exit 1
}

start_server

echo "[run] DATA_DIR=${DATA_DIR}"
echo "[run] TOOL_FILE=${TOOL_FILE}"
echo "[run] retrieval URL=${ONLINE_SEARCH_RETRIEVAL_URL} topk=${ONLINE_SEARCH_TOPK}"
bash examples/agent_loop/run_rollout_only_bs_sweep.sh
