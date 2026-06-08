#!/usr/bin/env bash
# Keep-alive loop: repeatedly run bs128 @ max_len=8192 rollout-only back-to-back
# for ~LOOP_HOURS hours (default ~12h). Each iteration is a fresh, independent
# rollout with a unique run id. A new iteration is only started while we are
# still inside the time budget, so total wall-clock lands close to LOOP_HOURS.
#
# Purpose: keep the container busy/alive; artifacts land under verl/logs.
#
# Override with env, e.g.:
#   LOOP_HOURS=11.8 bash examples/agent_loop/run_bs128_8192_keepalive_loop.sh
set -uo pipefail

LOOP_HOURS=${LOOP_HOURS:-11.8}
export LOG_ROOT=${LOG_ROOT:-/home/tiger/verl/logs}
REPO=/home/tiger/verl
cd "$REPO"
mkdir -p "$LOG_ROOT"

START=$(date +%s)
END=$(awk -v s="$START" -v h="$LOOP_HOURS" 'BEGIN{printf "%d", s + h*3600}')
LOOP_STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_ROOT}/bs128_8192_keepalive_loop_${LOOP_STAMP}.log"

{
  echo "=========================================================="
  echo "bs128 @ max_len=8192 keep-alive loop"
  echo "  start    : $(date)"
  echo "  deadline : $(date -d @"$END")  (~${LOOP_HOURS}h)"
  echo "  log_root : ${LOG_ROOT}"
  echo "  master   : ${MASTER_LOG}"
  echo "=========================================================="
} | tee -a "$MASTER_LOG"

i=0
while [ "$(date +%s)" -lt "$END" ]; do
    i=$((i + 1))
    now=$(date +%s); elapsed=$((now - START)); remain=$((END - now))
    {
      echo ""
      echo ">>>>> iteration ${i} | elapsed $((elapsed / 3600))h$(((elapsed % 3600) / 60))m | remaining $((remain / 3600))h$(((remain % 3600) / 60))m | $(date) <<<<<"
    } | tee -a "$MASTER_LOG"

    iter_tag="bs128loop_8192_${LOOP_STAMP}_iter${i}"
    env BS_GRID="128" \
        MAX_MODEL_LEN=8192 \
        HOTPOT_MAX_TOOL_CALLS=0 \
        HOTPOT_MIN_TOOL_CALLS=1 \
        LOG_ROOT="${LOG_ROOT}" \
        SWEEP_TAG="${iter_tag}" \
        bash examples/agent_loop/run_rollout_only_bs_sweep.sh >> "$MASTER_LOG" 2>&1
    st=$?
    echo ">>>>> iteration ${i} finished, exit=${st} <<<<<" | tee -a "$MASTER_LOG"

    # Belt-and-suspenders: make sure Ray/SGLang are down before the next round.
    ray stop --force >/dev/null 2>&1 || true
    sleep 5
done

total=$(( $(date +%s) - START ))
echo "" | tee -a "$MASTER_LOG"
echo "Loop complete: ${i} iterations, $((total / 3600))h$(((total % 3600) / 60))m elapsed." | tee -a "$MASTER_LOG"
