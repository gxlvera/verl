#!/usr/bin/env bash
# GRPO | Qwen3-1.7B | tool_agent loop on GSM8K | SGLang async rollout | FSDP
#
# First-stage rollout-profiling baseline for the draft-model tool-call prefetch
# project. Uses the @function_tool calc_gsm8k_reward defined in
# examples/agent_loop/gsm8k_function_tool.py.
#
# Prep (run once):
#   PYTHONPATH=$(pwd) python examples/data_preprocess/gsm8k_tool_agent_loop.py   # -> ~/data/gsm8k/{train,test}.parquet
#
# Run (this node has no GPU; launch on a GPU worker), e.g.:
#   mlx worker launch -- bash examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent.sh
#
# Knobs (env vars):
#   MODEL_PATH            HF id or local path        (default: Qwen/Qwen3-1.7B)
#   NGPUS_PER_NODE        GPUs on the worker         (default: 8)
#   ROLLOUT_TP            rollout tensor parallel    (default: 1)
#   TRAIN_BATCH_SIZE      prompts per step           (default: 16)
#   PPO_MICRO_BATCH_SIZE_PER_GPU actor update micro batch per GPU (default: 8)
#   ROLLOUT_N             samples per prompt          (default: 8)
#   DATA_SHUFFLE          shuffle prompt order        (default: False)
#   DATA_SEED             fixed prompt/data seed      (default: 42)
#   ROLLOUT_DO_SAMPLE     sample rollout tokens       (default: False)
#   ROLLOUT_TEMPERATURE   rollout temperature         (default: 0)
#   ROLLOUT_TOP_P         rollout top-p               (default: 1.0)
#   ROLLOUT_TOP_K         rollout top-k               (default: -1)
#   TOTAL_STEPS           training steps (profiling)  (default: 20)
#   GSM8K_TOOL_SLEEP_MS   fixed async tool latency ms (default: 300)
#   GSM8K_TOOL_SLEEP_DIST weighted async latency dist (e.g. "0:30,2000:70"; overrides fixed sleep)
#   GSM8K_TOOL_SLEEP_SEED seed for weighted latency sampling (default: unset)
#   GSM8K_MIN_TOOL_CALLS  auto-call tool until this count per sample (default: 2)
#   MOCK_BATCH_SIZE       completed samples per rollout timestamp (default: 32)
#   SGLANG_PROMETHEUS_ENABLE expose SGLang /metrics endpoint (default: false)
#   SGLANG_PROMETHEUS_PORT   Prometheus scrape port in Ray config (default: 9090)
#   LOG_ROOT              raw/tensorboard log root    (default: /root/logs)
#   LOG_TO_FILE           save full stdout/stderr     (default: 1)

set -xeuo pipefail

# huggingface.co is blocked on this infra; pull the model from ModelScope instead.
# verl patches huggingface_hub -> ModelScope when this is set (see verl/__init__.py).
# Set VERL_USE_MODELSCOPE=False if the worker has direct HF access or a local MODEL_PATH.
export VERL_USE_MODELSCOPE=${VERL_USE_MODELSCOPE:-True}

# Keep MODEL_PATH as the hub repo id (not a local dir) so the ModelScope patch can
# resolve it. Override with a local path if you've pre-downloaded the weights.
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
NNODES=${NNODES:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-False}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PER_TURN_MAX_RESPONSE_LENGTH=${PER_TURN_MAX_RESPONSE_LENGTH:-}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-}
MAX_TOOL_RESPONSE_LENGTH=${MAX_TOOL_RESPONSE_LENGTH:-4096}
DATA_SHUFFLE=${DATA_SHUFFLE:-False}
DATA_SEED=${DATA_SEED:-42}
ACTOR_SHUFFLE=${ACTOR_SHUFFLE:-False}
ACTOR_DATA_LOADER_SEED=${ACTOR_DATA_LOADER_SEED:-42}
DATA_ENABLE_THINKING=${DATA_ENABLE_THINKING:-False}

TOTAL_STEPS=${TOTAL_STEPS:-20}

# Expose tool latency knobs to the rollout workers. If neither is configured,
# or GSM8K_TOOL_SLEEP_MS=0, the tool injects no sleep.
export GSM8K_TOOL_SLEEP_MS=${GSM8K_TOOL_SLEEP_MS:-300}
export GSM8K_TOOL_SLEEP_DIST=${GSM8K_TOOL_SLEEP_DIST:-}
export GSM8K_TOOL_SLEEP_SEED=${GSM8K_TOOL_SLEEP_SEED:-}
export GSM8K_MIN_TOOL_CALLS=${GSM8K_MIN_TOOL_CALLS:-2}
export HOTPOT_TOOL_SLEEP_MS=${HOTPOT_TOOL_SLEEP_MS:-${GSM8K_TOOL_SLEEP_MS}}
export HOTPOT_TOOL_SLEEP_DIST=${HOTPOT_TOOL_SLEEP_DIST:-${GSM8K_TOOL_SLEEP_DIST}}
export HOTPOT_TOOL_SLEEP_SEED=${HOTPOT_TOOL_SLEEP_SEED:-${GSM8K_TOOL_SLEEP_SEED}}
export HOTPOT_TOOL_LATENCY_SECONDS_LIST=${HOTPOT_TOOL_LATENCY_SECONDS_LIST:-}
export HOTPOT_MIN_TOOL_CALLS=${HOTPOT_MIN_TOOL_CALLS:-${GSM8K_MIN_TOOL_CALLS}}
export HOTPOT_AUTO_TOOL_NAME=${HOTPOT_AUTO_TOOL_NAME:-}
export HOTPOT_SPECULATIVE_TOOL_PREFETCH=${HOTPOT_SPECULATIVE_TOOL_PREFETCH:-}
export HOTPOT_SPECULATIVE_JSONL=${HOTPOT_SPECULATIVE_JSONL:-}
export HOTPOT_MAIN_ENABLE_THINKING=${HOTPOT_MAIN_ENABLE_THINKING:-}
export HOTPOT_NON_THINKING_MAX_NEW_TOKENS=${HOTPOT_NON_THINKING_MAX_NEW_TOKENS:-}
export ONLINE_SEARCH_RETRIEVAL_URL=${ONLINE_SEARCH_RETRIEVAL_URL:-${ONLINE_SEARCH_URL:-}}
export ONLINE_SEARCH_TOPK=${ONLINE_SEARCH_TOPK:-3}
export ONLINE_SEARCH_TIMEOUT=${ONLINE_SEARCH_TIMEOUT:-20}
export ONLINE_SEARCH_MAX_CHARS=${ONLINE_SEARCH_MAX_CHARS:-${MAX_TOOL_RESPONSE_LENGTH}}
export AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH=${PER_TURN_MAX_RESPONSE_LENGTH}
export MOCK_BATCH_SIZE=${MOCK_BATCH_SIZE:-32}
SGLANG_PROMETHEUS_ENABLE=${SGLANG_PROMETHEUS_ENABLE:-false}
SGLANG_PROMETHEUS_PORT=${SGLANG_PROMETHEUS_PORT:-9090}

DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
TOOL_FILE=${TOOL_FILE:-examples/agent_loop/gsm8k_function_tool.py}

PROJECT_NAME=${PROJECT_NAME:-specTool}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_tool_agent_sglang_sleep${GSM8K_TOOL_SLEEP_MS}ms_toolcalls${GSM8K_MIN_TOOL_CALLS}_$(date +%Y%m%d_%H%M)}
export WANDB_ENTITY=${WANDB_ENTITY:-gxlvera-carnegie-mellon-university}

LOG_ROOT=${LOG_ROOT:-/root/logs}
LOG_TO_FILE=${LOG_TO_FILE:-1}
mkdir -p "${LOG_ROOT}"
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${LOG_ROOT}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}
if [[ "${LOG_TO_FILE}" == "1" ]]; then
    RUN_LOG_FILE=${RUN_LOG_FILE:-${LOG_ROOT}/${EXPERIMENT_NAME}.log}
    exec > >(tee -a "${RUN_LOG_FILE}") 2>&1
    echo "Saving full run log to ${RUN_LOG_FILE}"
fi
echo "Tool config: gsm8k_sleep_ms=${GSM8K_TOOL_SLEEP_MS}, hotpot_sleep_ms=${HOTPOT_TOOL_SLEEP_MS}, sleep_dist=${HOTPOT_TOOL_SLEEP_DIST:-${GSM8K_TOOL_SLEEP_DIST:-<unset>}}, latency_seconds_list=${HOTPOT_TOOL_LATENCY_SECONDS_LIST:-<unset>}, sleep_seed=${HOTPOT_TOOL_SLEEP_SEED:-${GSM8K_TOOL_SLEEP_SEED:-<unset>}}, min_tool_calls=${HOTPOT_MIN_TOOL_CALLS:-${GSM8K_MIN_TOOL_CALLS}}, speculative_prefetch=${HOTPOT_SPECULATIVE_TOOL_PREFETCH:-<unset>}, mock_batch_size=${MOCK_BATCH_SIZE}, per_turn_max_response=${PER_TURN_MAX_RESPONSE_LENGTH:-<unset>}, max_model_len=${MAX_MODEL_LEN:-<unset>}, max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}, sglang_prometheus=${SGLANG_PROMETHEUS_ENABLE}:${SGLANG_PROMETHEUS_PORT}"
echo "Determinism config: data_shuffle=${DATA_SHUFFLE}, data_seed=${DATA_SEED}, data_enable_thinking=${DATA_ENABLE_THINKING}, actor_shuffle=${ACTOR_SHUFFLE}, actor_data_loader_seed=${ACTOR_DATA_LOADER_SEED}, rollout_do_sample=${ROLLOUT_DO_SAMPLE}, temperature=${ROLLOUT_TEMPERATURE}, top_p=${ROLLOUT_TOP_P}, top_k=${ROLLOUT_TOP_K}"
echo "W&B run name: ${EXPERIMENT_NAME}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.shuffle="${DATA_SHUFFLE}" \
    data.seed="${DATA_SEED}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking="${DATA_ENABLE_THINKING}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.shuffle="${ACTOR_SHUFFLE}" \
    actor_rollout_ref.actor.data_loader_seed="${ACTOR_DATA_LOADER_SEED}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.do_sample="${ROLLOUT_DO_SAMPLE}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.function_tool_path="${TOOL_FILE}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.prometheus.enable="${SGLANG_PROMETHEUS_ENABLE}" \
    actor_rollout_ref.rollout.prometheus.port="${SGLANG_PROMETHEUS_PORT}" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-null}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GSM8K_TOOL_SLEEP_MS="'${GSM8K_TOOL_SLEEP_MS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GSM8K_TOOL_SLEEP_DIST="'${GSM8K_TOOL_SLEEP_DIST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GSM8K_TOOL_SLEEP_SEED="'${GSM8K_TOOL_SLEEP_SEED}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GSM8K_MIN_TOOL_CALLS="'${GSM8K_MIN_TOOL_CALLS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_TOOL_SLEEP_MS="'${HOTPOT_TOOL_SLEEP_MS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_TOOL_SLEEP_DIST="'${HOTPOT_TOOL_SLEEP_DIST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_TOOL_SLEEP_SEED="'${HOTPOT_TOOL_SLEEP_SEED}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_TOOL_LATENCY_SECONDS_LIST="'${HOTPOT_TOOL_LATENCY_SECONDS_LIST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_MIN_TOOL_CALLS="'${HOTPOT_MIN_TOOL_CALLS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_AUTO_TOOL_NAME="'${HOTPOT_AUTO_TOOL_NAME}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_SPECULATIVE_TOOL_PREFETCH="'${HOTPOT_SPECULATIVE_TOOL_PREFETCH}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_SPECULATIVE_JSONL="'${HOTPOT_SPECULATIVE_JSONL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_MAIN_ENABLE_THINKING="'${HOTPOT_MAIN_ENABLE_THINKING}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.HOTPOT_NON_THINKING_MAX_NEW_TOKENS="'${HOTPOT_NON_THINKING_MAX_NEW_TOKENS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ONLINE_SEARCH_RETRIEVAL_URL="'${ONLINE_SEARCH_RETRIEVAL_URL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ONLINE_SEARCH_TOPK="'${ONLINE_SEARCH_TOPK}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ONLINE_SEARCH_TIMEOUT="'${ONLINE_SEARCH_TIMEOUT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.ONLINE_SEARCH_MAX_CHARS="'${ONLINE_SEARCH_MAX_CHARS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH="'${AGENT_LOOP_PER_TURN_MAX_RESPONSE_LENGTH}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.MOCK_BATCH_SIZE="'${MOCK_BATCH_SIZE}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PARTIAL_ASYNC_NUM_STARTS="'${PARTIAL_ASYNC_NUM_STARTS:-}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PARTIAL_ASYNC_WARMUP_STEPS="'${PARTIAL_ASYNC_WARMUP_STEPS:-}'" \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "$@"
