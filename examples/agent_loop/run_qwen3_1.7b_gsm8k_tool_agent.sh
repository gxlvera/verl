#!/usr/bin/env bash
# GRPO | Qwen3-1.7B | tool_agent loop on GSM8K | vLLM async rollout | FSDP
#
# First-stage rollout-profiling baseline for the draft-model tool-call prefetch
# project. Uses the @function_tool calc_gsm8k_reward defined in
# examples/agent_loop/gsm8k_function_tool.py.
#
# Prep (run once):
#   python examples/data_preprocess/gsm8k_tool_agent_loop.py   # -> ~/data/gsm8k/{train,test}.parquet
#
# Run (this node has no GPU; launch on a GPU worker), e.g.:
#   mlx worker launch -- bash examples/agent_loop/run_qwen3_1.7b_gsm8k_tool_agent.sh
#
# Knobs (env vars):
#   MODEL_PATH            HF id or local path        (default: Qwen/Qwen3-1.7B)
#   NGPUS_PER_NODE        GPUs on the worker         (default: 8)
#   ROLLOUT_TP            rollout tensor parallel    (default: 1)
#   TRAIN_BATCH_SIZE      prompts per step           (default: 32)
#   ROLLOUT_N             samples per prompt          (default: 8)
#   TOTAL_STEPS           training steps (profiling)  (default: 3)
#   GSM8K_TOOL_SLEEP_MS   inject tool latency (ms)    (default: 0)

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

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

TOTAL_STEPS=${TOTAL_STEPS:-3}

# Expose the per-call tool latency knob to the rollout workers.
export GSM8K_TOOL_SLEEP_MS=${GSM8K_TOOL_SLEEP_MS:-0}

DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k}
TOOL_FILE=${TOOL_FILE:-examples/agent_loop/gsm8k_function_tool.py}

PROJECT_NAME=${PROJECT_NAME:-verl_tool_agent_gsm8k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_tool_agent_vllm_$(date +%Y%m%d_%H%M)}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.function_tool_path="${TOOL_FILE}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "$@"
