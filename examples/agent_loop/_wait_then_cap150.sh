#!/usr/bin/env bash
# Wait for extra2 set to finish (sentinel EXTRA2_DONE + GPU idle), then launch the
# cap150 full-shadow run. Uses nvidia-smi for idleness (avoids ps self-match).
set -uo pipefail
cd /home/tiger/verl

# 1) wait for extra2 orchestrator sentinel
while ! grep -q EXTRA2_DONE logs/extra2_orchestrator.log 2>/dev/null; do sleep 15; done

# 2) wait until GPUs are idle (max util across GPUs < 5% for 3 consecutive checks)
idle=0
while [ "$idle" -lt 3 ]; do
    maxu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    if [ "${maxu:-100}" -lt 5 ]; then idle=$((idle+1)); else idle=0; fi
    sleep 10
done
sleep 10

bash examples/agent_loop/run_real_bm25_latency_fullshadow_cap150_bs256.sh
